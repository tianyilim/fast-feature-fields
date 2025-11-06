# Before this: install:
# pip install onnx onnxsim onnxruntime onnxscript

import sys
import torch
from pathlib import Path
import onnx
import onnxsim
from torch.export.dynamic_shapes import Dim

from f3 import init_event_model, load_weights_ckpt
from f3.utils import (setup_torch, ev_to_frames, plot_patched_features,
                      smooth_time_weighted_rgb_encoding, BaseExtractor)
from f3.tasks.depth.utils import init_depth_model, load_depth_weights, get_disparity_image
from f3.tasks.optical_flow.utils import init_flow_model, load_flow_weights, flow_viz_np
from f3.tasks.segmentation.utils import init_segmentation_model, load_segmentation_weights, cityscapes_palette

setup_torch(cudnn_benchmark=False)

################ Path where Models are downloaded ################
BASE_F3_MODEL_PATH = "pretrained_models/f3/"
################ Model Names to Load ################
F3_MODEL_NAME = "patchff_fullcardaym3ed_small_20ms.pth"
F3_CONFIG_NAME = "confs/ff/modeloptions/1280x720x20_patchff_ds1_small.yml"
################ Path to a M3ED Sequence ################
H5PATH = "data/car_urban_day_penno_small_loop/car_urban_day_penno_small_loop_data.h5"
TSPATH = "data/car_urban_day_penno_small_loop/50khz_car_urban_day_penno_small_loop_data.npy"

# Data Loader
extractor = BaseExtractor(H5PATH, TSPATH, w=1280, h=720, time_ctx=20000,
                          time_pred=20000, bucket=1000, max_numevents_ctx=800000,
                          randomize_ctx=False, camera="left")

# Load an example of F3 head
eventff_model = init_event_model(F3_CONFIG_NAME, return_feat=True, return_logits=False).cuda()
eventff_model_compilied = torch.compile(
    eventff_model,
    fullgraph=False,
    backend="inductor",
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
    },
)
# Load from compiled model, but we want to export the raw model instead.
eventff_model_compilied.load_state_dict(
            torch.load(Path(BASE_F3_MODEL_PATH) / F3_MODEL_NAME, map_location="cpu", weights_only=True)['model'])
eventff_model.cuda().eval()
print(f"Loaded F3 model ckpt from {F3_MODEL_NAME}")

# Get example data for tracing
IMG_IDX = 700
t0 = extractor.hdf5_file["ovc/ts"][IMG_IDX]
# get events in fixed time window
ctx, totcnt = extractor.get_ctx_fixedtime(t0)    
ctx, totcnt = ctx.cuda(), torch.tensor([totcnt]).cuda()

# Forward sanity check to ensure all is well with input model
print("FORWARD SANITY CHECK")
print(ctx.shape, totcnt)
eventff_out = eventff_model(ctx, totcnt)
print(eventff_out.shape)
# Input shape: N, 4

# Export!
out_file = Path(F3_CONFIG_NAME).stem + ".onnx"
out_file = Path(out_file)

torch.onnx.export(eventff_model, (ctx, totcnt),
                  f=out_file,
                  input_names=["currentBlock", "eventCounts"],
                  output_names=["feats"],
                  dynamic_shapes={'currentBlock': {0: Dim.DYNAMIC, 1: Dim.STATIC}, 'eventCounts':{0: Dim.DYNAMIC}},
                  dynamo=True,
                  verbose=False,
                  opset_version=17
                 )
print(f"Exported to file [{out_file}]")

# Check if saving to ONNX worked
assert out_file.exists()
onnx_check = onnx.load(out_file)
onnx.checker.check_model(onnx_check)
print("ONNX model check passed.")

# Simplify model
print("Simplifying model...")
model_sim, check = onnxsim.simplify(onnx_check)
assert check
onnx.save(model_sim, out_file)
print(f"Saved simplified ONNX model to [{out_file}]")
