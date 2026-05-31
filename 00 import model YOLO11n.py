from ultralytics import YOLO


exit()
import onnx
import onnx.numpy_helper as nph
from onnx import helper
# Load a COCO-pretrained YOLO11n model

model = YOLO("yolo11s.pt")
# Export the model
#model.export(format="onnx",opset=11,optimize=True,simplify=True,imgsz=640,dynamic=False)
model.export(format="onnx",optimize=True,simplify=True,imgsz=640,dynamic=False)

#yolo export model=yolov8s.pt imgsz=320 format=onnx opset=11 optimize=True simplify=true
model_name='yolo11s'
model_onnx = onnx.load(f"./{model_name}.onnx")
#model_name='yolo11n_480'
graph = model_onnx.graph
# Знайти всі Concat ноди
for node in graph.node:
    if node.op_type == "Concat":
        print(f"Concat: inputs={list(node.input)}, output={list(node.output)}")

#Concat: inputs=['/model.23/Mul_2_output_0' (4x8400), '/model.23/Sigmoid_output_0' (80x8400)], output=['output0']


for node in graph.node:
    if node.op_type == "Concat":
        for out in node.output:
            if out == "output0":  # або як називається фінальний вихід
                final_concat = node
                concat_inputs = list(node.input)
                print(f"Знайдено: {concat_inputs}")

# Отримати shape інформацію для нових виходів
shape_info = {vi.name: vi for vi in graph.value_info}
shape_info.update({o.name: o for o in graph.output})

# Замінити один вихід на два окремих
new_outputs = []
output_names = ["boxes", "scores"]  # нові імена

for i, inp_name in enumerate(concat_inputs):
    # Знайти shape тензора
    if inp_name in shape_info:
        vi = shape_info[inp_name]
    else:
        # Створити базовий ValueInfo якщо немає
        vi = helper.make_tensor_value_info(inp_name, onnx.TensorProto.FLOAT, None)
    
    # Перейменувати для зручності
    new_vi = helper.make_tensor_value_info(
        output_names[i],
        onnx.TensorProto.FLOAT,
        None
    )
    new_outputs.append(new_vi)

    # Перенаправити вихід — знайти ноду що продукує цей тензор
    # і додати alias якщо потрібно
    for node in graph.node:
        for j, out in enumerate(node.output):
            if out == inp_name:
                node.output[j] = output_names[i]
                print(f"  Перенаправлено: {inp_name} → {output_names[i]}")

# Видалити фінальний Concat з графу
graph.node.remove(final_concat)

# Замінити виходи моделі
while len(graph.output) > 0:
    graph.output.pop()

graph.output.extend(new_outputs)

# Зберегти
#onnx.checker.check_model(model_onnx)
onnx.save(model_onnx, f"./models/{model_name}_split.onnx")
print("Збережено split модель!")

model_onnx = onnx.load(f"./models/{model_name}_split.onnx")
for node in graph.node:
    if node.op_type == "Concat":
        print(f"Concat: inputs={list(node.input)}, output={list(node.output)}")



import onnxruntime as ort

# Load the session
#session = ort.InferenceSession("yolo11x_split.onnx")
session = ort.InferenceSession(f"./models/{model_name}_split.onnx")
# Get output details
outputs = session.get_outputs()
for output in outputs:
    print(f"Name: {output.name}")
    print(f"Shape: {output.shape}")
    print(f"Type: {output.type}")