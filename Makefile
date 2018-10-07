PYTHON=python3

# VARIABLES

NUM_PHASES=500
MAP_CHANNEL_WEIGHT=3500
PLOT_INTERVAL=5
SAVE_INTERVAL=50

# COMMANDS TO GENERATE TINY IMAGES

monet_tiny:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_tiny/monet.png \
	              --input-map-file data/samples_tiny/monet_style.png \
	              --output-map-file data/samples_tiny/monet_out_style.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

renoir_tiny:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_tiny/renoir.png \
	              --input-map-file data/samples_tiny/renoir_style.png \
	              --output-map-file data/samples_tiny/renoir_out_style.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

renoir_2_tiny:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_tiny/renoir.png \
	              --input-map-file data/samples_tiny/renoir_style.png \
	              --output-map-file data/samples_tiny/renoir_out_style_2.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

freddie_tiny:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_tiny/bergeron.png \
	              --input-map-file data/samples_tiny/bergeron_map.png \
	              --output-content-file data/samples_tiny/freddie.png \
	              --output-map-file data/samples_tiny/freddie_map.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

# COMMANDS TO GENERATE SMALL IMAGES

monet_small:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_small/monet.png \
	              --input-map-file data/samples_small/monet_style.png \
	              --output-map-file data/samples_small/monet_out_style.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

renoir_small:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_small/renoir.png \
	              --input-map-file data/samples_small/renoir_style.png \
	              --output-map-file data/samples_small/renoir_out_style.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

renoir_2_small:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_small/renoir.png \
	              --input-map-file data/samples_small/renoir_style.png \
	              --output-map-file data/samples_small/renoir_out_style_2.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

freddie_small:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples_small/bergeron.png \
	              --input-map-file data/samples_small/bergeron_map.png \
	              --output-content-file data/samples_small/freddie.png \
	              --output-map-file data/samples_small/freddie_map.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

# COMMANDS TO GENERATE FULL IMAGES

monet_full:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples/monet.png \
	              --input-map-file data/samples/monet_style.png \
	              --output-map-file data/samples/monet_out_style.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

renoir_full:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples/renoir.png \
	              --input-map-file data/samples/renoir_style.png \
	              --output-map-file data/samples/renoir_out_style.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

renoir_2_full:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples/renoir.png \
	              --input-map-file data/samples/renoir_style.png \
	              --output-map-file data/samples/renoir_out_style_2.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)

freddie_full:
	$(PYTHON) src/generate.py \
	              --input-style-file data/samples/bergeron.png \
	              --input-map-file data/samples/bergeron_map.png \
	              --output-content-file data/samples/freddie.png \
	              --output-map-file data/samples/freddie_map.png \
	              --device cuda \
	              --num-phases $(NUM_PHASES) \
	              --map-channel-weight $(MAP_CHANNEL_WEIGHT) \
	              --save-interval $(SAVE_INTERVAL) \
	              --plot-interval $(PLOT_INTERVAL)
