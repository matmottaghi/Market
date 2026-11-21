# توضیح فایل: بررسی قرارداد واقعی توقف Keras و بازیابی وزن منتخب هنگام بهبود کمتر از min_delta؛ بدون آموزش یا شبکه.
# test_minimum_loss_can_differ_from_restored_epoch نشان می‌دهد چرا نشانگر نمودار باید از best_epoch خود callback خوانده شود.
def test_minimum_loss_can_differ_from_restored_epoch():
    import keras
    import numpy as np

    class Weights:
        stop_training=False
        value=0
        def get_weights(self):return [np.array([self.value])]
        def set_weights(self,weights):self.value=int(weights[0][0])

    model=Weights()
    stopping=keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,min_delta=.001,restore_best_weights=True)
    stopping.set_model(model)
    stopping.on_train_begin()
    losses=[1.,.9995]+[1.1]*11
    for epoch,loss in enumerate(losses):
        model.value=epoch
        stopping.on_epoch_end(epoch,{'val_loss':loss})
    assert model.stop_training and stopping.stopped_epoch==12
    assert stopping.best_epoch==0 and int(np.argmin(losses))==1
    stopping.on_train_end()
    assert model.value==0
