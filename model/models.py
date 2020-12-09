import time

from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import DummyVecEnv

from env.multiple_stock_trade import StockEnvTrade
from env.multiple_stock_train import StockEnvTrain
from env.multiple_stock_validation import StockEnvValidation
from preprocessing.preprocessors import *


def train_a2c(env_train, model_name, time_steps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=time_steps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_ddpg(env_train, model_name, time_steps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=time_steps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end - start) / 60, ' minutes')
    return model


def train_ppo(env_train, model_name, time_steps=50000):
    """PPO model"""
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8)
    model.learn(total_timesteps=time_steps)
    end = time.time()
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def drl_prediction(df, model, name, last_state, iter_num, unique_trade_date, rebalance_window, turbulence_threshold,
                   initial):
    """Make a prediction based on trained model."""
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data, turbulence_threshold=turbulence_threshold,
                                                   initial=initial, previous_state=last_state, model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    """validation process"""
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    """Calculate Sharpe ratio based on validation results"""
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe


def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []

    # based on the analysis of the in-sample data
    # turbulence_threshold = 140
    insample_turbulence = df[(df.datadate < 20151000) & (df.datadate >= 20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        # initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[
            df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]
            ].to_list()[-1]

        start_date_index = end_date_index - validation_window * 30 + 1
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        # training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        # validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv(
            [lambda: StockEnvValidation(validation, turbulence_threshold=turbulence_threshold, iteration=i)]
        )
        obs_val = env_val.reset()

        print("==> Model training from: ", 20090000, "to ", unique_trade_date[i - rebalance_window - validation_window])
        print("======PPO Training========")
        model_ppo = train_ppo(env_train, model_name="PPO_100k_dow_{}".format(i), time_steps=100000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        ppo_sharpe_list.append(sharpe_ppo)
        model_ensemble = model_ppo

        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        last_state_ensemble = drl_prediction(
            df=df,
            model=model_ensemble,
            name="ensemble",
            last_state=last_state_ensemble,
            iter_num=i,
            unique_trade_date=unique_trade_date,
            rebalance_window=rebalance_window,
            turbulence_threshold=turbulence_threshold,
            initial=initial
        )
        print("============Trading Done============")

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
