import os

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
'''cd bin
Unblock-File ./API/bin/envsetup.ps1
./API/bin/envsetup.ps1'''
#https://docs.qualcomm.com/doc/80-90441-15/topic/appx-export-yolov8.html

# python ./API/bin/x86_64-windows-msvc/snpe-onnx-to-dlc -i ./models/yolo11n.onnx -o ./models/yolo11n.dlc
# ./API/bin/x86_64-windows-msvc/snpe-dlc-quant.exe --input_dlc ./models/yolo11n.dlc --input_list 02_calibrate_dataset_list.txt --output_dlc ./models/yolo11n_f32_to_w8a8.dlc

#os.environ['QAIRT_SDK_ROOT'] = 'C:/Users/dmytr/Desktop/AIBuilder/API'

# Add the QAIRT lib directory explicitly
#os.environ["PATH"] += r";C:\Users\dmytr\Desktop\AIBuilder\API\lib\x86_64-windows-msvc"


import json
import os
import numpy as np
import qairt
from qairt import Device, DevicePlatformType
from qairt.api.converter.converter_config import CalibrationConfig
# Try 3
from qairt.api.compiler.config_util import HtpGraphConfig


model_name = "yolo11s"

# Step 4: Convert the model using calibration data
# Initialize the calibration config (Note: Weight, bias, and activation values default to 8-bit)
print("Converting the onnx model to a QAIRT model using calibration data...")
calibration_config = CalibrationConfig(dataset= "02_calibrate_dataset_list.txt",
                                       per_channel_quantization=True,
                                       batch_size=1,
                                       
                                       )

converted_model: qairt.Model = qairt.convert(
                f"./models/{model_name}_split.onnx",
                calibration_config=calibration_config
                #output_path=f"./models/{model_name}_f32.dlc"
                ,backend="HTP"


            )

print("Model conversion complete!")
print(f"Converted model input tensors: {converted_model.input_tensors}")
print(f"Converted model output tensors: {converted_model.output_tensors}")
print(f"Converted model name: {converted_model.name}")
converted_model.save(f"./models/{model_name}_w8a8.dlc")


# Step 5: Compile the model for HTP backend
print("Compiling the model for HTP backend...")

htp_graph_config = HtpGraphConfig(name=converted_model.name,
                              vtcm_size_in_mb=0)
# Compile the model for a particular SOC  
compile_config =  qairt.CompileConfig(backend="HTP"
                                      , soc_details="chipset:QCS6490"
                               ,graph_custom_configs=[htp_graph_config]
                                     )  


#print(qairt.CompileConfig.model_fields)  # Pydantic v2

compiled_model: qairt.CompiledModel = qairt.compile(converted_model
                                                    ,config=compile_config                                       
                                                    
                                                    )



print("Model compilation complete!")
print(f"compiled_model input tensors: {compiled_model.input_tensors}")
print(f"compiled_model output tensors: {compiled_model.output_tensors}")

# Step 2: Preprocess the image
print("Running QAIRT inference...")

# Step 9: Save the compiled model
compiled_model.save(f"./models/{model_name}.bin")

# Step 10: Save the compiled model info as a json
with open(f"{model_name}_info.json", "w") as f:
    json.dump(compiled_model.module.info.as_dict(), f, indent=4)




#./API/bin/x86_64-windows-msvc/qnn-context-binary-generator --model ./API/lib/x86_64-windows-msvc/QnnModelDlc.dll --backend ./API/lib/x86_64-windows-msvc/QnnHtp.dll --dlc_path ./models/yolo11n_w8a8.dlc --output_dir ./models --binary_file yolo11n --config_file config_file.json