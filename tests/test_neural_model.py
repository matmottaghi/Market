# توضیح فایل: آزمون واقعی مدل Keras روی Python سراسری؛ نصب کتابخانه انجام نمی‌شود.
# تابع test_mask_training_and_saved_model ماسک padding، یک گام یادگیری و بازخوانی فایل keras را بررسی می‌کند.
import numpy as np


def test_mask_training_and_saved_model(tmp_path):
    import keras
    from forecast.neural_network import build_network
    model = build_network(8)
    x = np.random.default_rng(1).normal(size=(2, 6, 8)).astype('float32')
    mask = np.array([[True, True, True, False, False, False]]*2)
    changed = x.copy()
    changed[:, 3:] = 999
    np.testing.assert_allclose(model([x, mask], training=False), model([changed, mask], training=False), atol=1e-6)
    scores = model.train_on_batch([x, mask], np.zeros((2, 3), dtype='float32'))
    assert np.isfinite(scores).all()
    destination = tmp_path/'network.keras'
    model.save(destination)
    restored = keras.models.load_model(destination)
    np.testing.assert_allclose(model([x, mask], training=False), restored([x, mask], training=False), atol=1e-6)
