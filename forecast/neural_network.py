# توضیح فایل: معماری عصبی LSTM و توابع زیان قابل ذخیره در مدل Keras؛ فقط هنگام آموزش یا بارگذاری مدل وارد شود.
# توابع: gaussian_nll ترکیب NLL نرمال و BCE جهت با وزن ۰٫۲؛ standardized_mae خطای میانگین؛ build_network ساخت دو LSTM با ماسک مستقل.
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras


@keras.saving.register_keras_serializable(package='Market')
def gaussian_nll(y_true, y_pred):
    mu = y_pred[:, :3]
    sigma = keras.ops.softplus(y_pred[:, 3:]) + .05
    nll = keras.ops.log(sigma) + .5*keras.ops.square((y_true-mu)/sigma)
    probability = keras.ops.clip(.5*(1+keras.ops.erf(mu/(sigma*2**.5))), 1e-6, 1-1e-6)
    target = keras.ops.cast(y_true > 0, 'float32')
    direction = -(target*keras.ops.log(probability)+(1-target)*keras.ops.log(1-probability))
    return keras.ops.mean(nll + .2*direction, axis=-1)


@keras.saving.register_keras_serializable(package='Market')
def standardized_mae(y_true, y_pred):
    return keras.ops.mean(keras.ops.abs(y_true-y_pred[:, :3]), axis=-1)


def build_network(feature_count, seed=42):
    keras.utils.set_random_seed(seed)
    values = keras.Input(shape=(None, feature_count), name='hourly_features')
    mask = keras.Input(shape=(None,), dtype='bool', name='valid_bars')
    hidden = keras.layers.LSTM(96, return_sequences=True, name='hourly_patterns')(values, mask=mask)
    hidden = keras.layers.Dropout(.20)(hidden)
    hidden = keras.layers.LSTM(48, name='session_context')(hidden, mask=mask)
    hidden = keras.layers.Dense(64, activation='swish', kernel_regularizer=keras.regularizers.l2(.0001))(hidden)
    hidden = keras.layers.Dropout(.25)(hidden)
    hidden = keras.layers.Dense(32, activation='swish')(hidden)
    output = keras.layers.Dense(6, name='log_return_distribution')(hidden)
    model = keras.Model([values, mask], output, name='DeepLSTM_21_sessions')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0005, clipnorm=1),
                  loss=gaussian_nll, metrics=[standardized_mae])
    return model
