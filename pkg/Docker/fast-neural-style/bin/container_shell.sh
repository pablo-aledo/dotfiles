INPUT_DIR=$(pwd)/input
mkdir -p $INPUT_DIR
echo "mounting input dir in "$INPUT_DIR

OUTPUT_DIR=$(pwd)/output
mkdir -p $OUTPUT_DIR
echo "mounting output dir in "$OUTPUT_DIR

docker run -it --rm -v $INPUT_DIR:/input -v $OUTPUT_DIR:/output neural_style_transfer:latest /bin/bash

# Apply style transfer
# th fast_neural_style.lua -model models/eccv16/starry_night.t7 -input_image /input/hoovertowernight.jpg -output_image /output/out.png

# CPU training
# source .env/bin/activate ; python scripts/make_style_dataset.py --train_dir /input/train --val_dir /input/val --output_file /input/train_data.h5
# th train.lua -h5_file /input/train_data.h5 -num_iterations 11 -checkpoint_every 10 -style_image /input/train/dali.jpg -style_image_size 384 -content_weights 1.0 -style_weights 5.0 -checkpoint_name /input/checkpoint -gpu -1
