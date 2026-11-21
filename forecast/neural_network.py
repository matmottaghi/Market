# توضیح فایل: معماری عصبی LSTM و توابع زیان قابل ذخیره در مدل Keras؛ فقط هنگام آموزش یا بارگذاری مدل وارد شود.
# توابع: gaussian_nll ترکیب NLL نرمال و BCE جهت با وزن ۰٫۲؛ standardized_mae خطای میانگین؛ build_network ساخت دو LSTM با ماسک مستقل.
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras


@keras.saving.register_keras_serializable(package='Market')
def gaussian_nll(y_true, y_pred):
    count=y_pred.shape[-1]//2
    mu = y_pred[:, :count]
    sigma = keras.ops.softplus(y_pred[:, count:]) + .05
    mask=y_true[:, count:] if y_true.shape[-1]==count*2 else keras.ops.ones_like(mu)
    y_true=y_true[:,:count]
    nll = keras.ops.log(sigma) + .5*keras.ops.square((y_true-mu)/sigma)
    probability = keras.ops.clip(.5*(1+keras.ops.erf(mu/(sigma*2**.5))), 1e-6, 1-1e-6)
    target = keras.ops.cast(y_true > 0, 'float32')
    direction = -(target*keras.ops.log(probability)+(1-target)*keras.ops.log(1-probability))
    return keras.ops.sum((nll+.2*direction)*mask,axis=-1)/keras.ops.maximum(keras.ops.sum(mask,axis=-1),1)


@keras.saving.register_keras_serializable(package='Market')
def standardized_mae(y_true, y_pred):
    count=y_pred.shape[-1]//2
    mask=y_true[:,count:] if y_true.shape[-1]==count*2 else keras.ops.ones_like(y_true)
    return keras.ops.sum(keras.ops.abs(y_true[:,:count]-y_pred[:,:count])*mask,axis=-1)/keras.ops.maximum(keras.ops.sum(mask,axis=-1),1)



def build_network(feature_count, seed=42, horizon_count=3):
    keras.utils.set_random_seed(seed)
    values = keras.Input(shape=(None, feature_count), name='hourly_features')
    mask = keras.Input(shape=(None,), dtype='bool', name='valid_bars')
    hidden = keras.layers.LSTM(96, return_sequences=True, name='hourly_patterns')(values, mask=mask)
    hidden = keras.layers.Dropout(.20)(hidden)
    hidden = keras.layers.LSTM(48, name='session_context')(hidden, mask=mask)
    hidden = keras.layers.Dense(64, activation='swish', kernel_regularizer=keras.regularizers.l2(.0001))(hidden)
    hidden = keras.layers.Dropout(.25)(hidden)
    hidden = keras.layers.Dense(32, activation='swish')(hidden)
    output = keras.layers.Dense(horizon_count, name='signed_hourly_return')(hidden)
    model = keras.Model([values, mask], output, name='DeepLSTM_21_sessions')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0005, clipnorm=1),
                  loss=return_loss, metrics=[return_mae])
    return model


@keras.saving.register_keras_serializable(package='Market')
def return_loss(y_true, y_pred):
    count=y_pred.shape[-1]
    target=y_true[:,:count];mask=y_true[:,count:] if y_true.shape[-1]==2*count else keras.ops.ones_like(y_pred)
    # Preserve the direction weight while learning signed returns directly.
    direction=keras.ops.binary_crossentropy(keras.ops.cast(target>0,'float32'),keras.ops.sigmoid(y_pred))
    error=keras.ops.square(target-y_pred)+.2*direction
    return keras.ops.sum(error*mask,axis=-1)/keras.ops.maximum(keras.ops.sum(mask,axis=-1),1)

@keras.saving.register_keras_serializable(package='Market')
def return_mae(y_true,y_pred):
    count=y_pred.shape[-1];mask=y_true[:,count:] if y_true.shape[-1]==2*count else keras.ops.ones_like(y_pred)
    return keras.ops.sum(keras.ops.abs(y_true[:,:count]-y_pred)*mask,axis=-1)/keras.ops.maximum(keras.ops.sum(mask,axis=-1),1)
