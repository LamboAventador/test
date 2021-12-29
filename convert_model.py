from onnx_tf.backend import prepare
import onnx

TF_PATH = "." # where the representation of tensorflow model will be stored
ONNX_PATH = "logo27_mobileNetv2_100_224_torch.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)
