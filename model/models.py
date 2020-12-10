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


def drl_prediction(df, model, name, last_state, turbulence_threshold):
    """Make a prediction based on trained model."""
    trade_data = data_split(df, start=20160102, end=2021010)
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data, turbulence_threshold=turbulence_threshold,
                                                   previous_state=last_state, model_name=name)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}.csv'.format(name), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    """validation process"""
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe():
    """Calculate Sharpe ratio based on validation results"""
    df_total_value = pd.read_csv('results/account_value_validation.csv', index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe


def run_ensemble_strategy(df, unique_trade_date):
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    ppo_sharpe_list = []
    last_state_ensemble = []

    # based on the analysis of the in-sample data
    # turbulence_threshold = 140
    insample_turbulence = df[(df.datadate < 20151000) & (df.datadate >= 20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()

    # Tuning trubulence index based on historical data
    # Turbulence lookback window is one quarter

    historical_turbulence = df.drop_duplicates(subset=['datadate'])
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
    train = data_split(df, start=20090000, end=20151001)
    env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

    # validation env
    validation = data_split(df, start=20151002, end=20160101)

    env_val = DummyVecEnv(
        [lambda: StockEnvValidation(validation, turbulence_threshold=turbulence_threshold)]
    )
    obs_val = env_val.reset()

    print("======PPO Training========")
    model_ppo = train_ppo(env_train, model_name="PPO_100k_dow", time_steps=100000)
    DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
    sharpe_ppo = get_validation_sharpe()
    print("PPO Sharpe Ratio: ", sharpe_ppo)

    ppo_sharpe_list.append(sharpe_ppo)
    model_ensemble = model_ppo

    last_state_ensemble = drl_prediction(df=df, model=model_ensemble, name="ensemble", last_state=last_state_ensemble,
                                         turbulence_threshold=turbulence_threshold)

    print("============Trading Done============")

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
