from InterfaceContoroller import InterfaceContoroller

def main(input_size=128, emotion_th=1, model_type="float"):
    ic = InterfaceContoroller()
    ic.capture(input_size=input_size, emotion_th=emotion_th, model_type=model_type)

if __name__ == "__main__":
    main(192, 1, "float")