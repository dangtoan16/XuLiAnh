import cv2 as cv
import numpy as np
import tensorflow as tf

# Load model
model_architecture = 'module/nhan_dien_chu_so_viet_tay_mnist/digit_config.json'
model_weights = 'module/nhan_dien_chu_so_viet_tay_mnist/digit_weight.h5'
model = tf.keras.models.model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

def predict_digits(img):
    color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    _, temp = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Phân đoạn các ký tự
    dem, label = cv.connectedComponents(temp)

    a = np.zeros(dem, np.int32)
    M, N = label.shape
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] += 1

    max_count = max(a[1:])
    nguong = max_count // 10
    margin = 5

    for r in range(1, dem):
        if a[r] > nguong:
            xmin = M - 1
            ymin = N - 1
            xmax = 0
            ymax = 0
            for x in range(0, M):
                for y in range(0, N):
                    if label[x, y] == r:
                        if x < xmin:
                            xmin = x
                        if y < ymin:
                            ymin = y
                        if x > xmax:
                            xmax = x
                        if y > ymax:
                            ymax = y

            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(M - 1, xmax + margin)
            ymax = min(N - 1, ymax + margin)

            # Predict digit
            word = temp[xmin:xmax+1, ymin:ymax+1]
            m, n = word.shape
            if m > n:
                word_vuong = np.zeros((m, m), np.uint8)
                word_vuong[0:m, 0:n] = word
            elif n > m:
                word_vuong = np.zeros((n, n), np.uint8)
                word_vuong[0:m, 0:n] = word
            else:
                word_vuong = word.copy()

            word_vuong = cv.resize(word_vuong, (20, 20))
            word = np.zeros((28, 28), np.uint8)
            word[0:20, 0:20] = word_vuong

            moments = cv.moments(word)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            xc = 14
            yc = 14
            word_vuong = np.zeros((28, 28), np.uint8)
            for x in range(0, 28):
                for y in range(0, 28):
                    r = word[x, y]
                    if r > 0:
                        dy = xc - cx
                        dx = yc - cy
                        x_moi = x + dx
                        y_moi = y + dy
                        if x_moi < 0:
                            x_moi = 0
                        if x_moi > 28 - 1:
                            x_moi = 28 - 1
                        if y_moi < 0:
                            y_moi = 0
                        if y_moi > 28 - 1:
                            y_moi = 28 - 1
                        word_vuong[x_moi, y_moi] = r

            sample = word_vuong / 255.0
            sample = sample.astype('float32')
            sample = np.expand_dims(sample, axis=0)
            sample = np.expand_dims(sample, axis=3)
            ket_qua = model.predict(sample, verbose=0)
            chu_so = np.argmax(ket_qua[0])

            # Draw bounding box and label
            cv.rectangle(color_img, (ymin, xmin), (ymax, xmax), (0, 255, 0), 2)
            cv.putText(color_img, str(chu_so), (ymin, xmin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return color_img