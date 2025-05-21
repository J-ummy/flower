model = keras.models.load_model('flower.keras')


path = "./images/tulips"

kind = {0: "daisy", 1: "dandelion", 2: "rose", 3: "sunflowers", 4: "tulips"}
for i , file in enumerate(os.listdir(path)):
    full = os.path.join(path, file)
    img = cv2.imdecode(
        np.fromfile(full, dtype = np.uint8),
        cv2.IMREAD_COLOR
    )[:, :, :: -1].copy()
    x = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    out = model.predict(x)
    idx = np.argmax(out[0])

    name = kind[idx]
    ax = plt.subplot(2, 5, i + 1)
    ax.set_title(name)
    ax.imshow(img)
    ax.axis("off")
plt.show()
