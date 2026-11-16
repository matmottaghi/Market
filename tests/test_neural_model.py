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


def test_504_heads_missing_labels_and_reload(tmp_path):
    """Unobserved targets must not change loss or gradients; full saved model reproduces every head."""
    import keras
    from forecast.neural_network import build_network,gaussian_nll
    model=build_network(15,horizon_count=504)
    x=np.random.default_rng(8).normal(size=(2,7,15)).astype('float32')
    valid=np.ones((2,7),bool)
    labels=np.zeros((2,1008),dtype='float32');labels[:,504+9:504+13]=1
    changed=labels.copy();changed[:,:9]=999
    raw=model([x,valid],training=False)
    assert raw.shape==(2,1008)
    np.testing.assert_allclose(gaussian_nll(labels,raw),gaussian_nll(changed,raw))
    assert np.isfinite(model.train_on_batch([x,valid],labels)).all()
    destination=tmp_path/'hourly.keras';model.save(destination)
    loaded=keras.models.load_model(destination)
    np.testing.assert_allclose(model([x,valid],training=False),loaded([x,valid],training=False),atol=1e-6)
