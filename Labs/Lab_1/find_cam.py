import cv2

print("Пошук доступних камер...")

working_index = None
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Камеру знайдено за індексом: {i}!")
            working_index = i
            cap.release()
            break
        cap.release()

if working_index is None:
    print("Жодної робочої камери не знайдено.")
    print("Перевірте: Параметри Windows -> Конфіденційність -> Камера -> Дозволити програмам доступ.")
else:
    print(f"\nЗапуск тесту для камери #{working_index} (Натисніть 'q' на клавіатурі для виходу)...")
    cap = cv2.VideoCapture(working_index, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Test Camera {working_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()