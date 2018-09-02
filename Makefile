PYTHON=python3
PLOT_INTERVALS=5

monet_tiny: data/output/monet_tiny.png
monet_small: data/output/monet_small.png
monet_full: data/output/monet.png
renoir_tiny: data/output/renoir_tiny.png
renoir_small: data/output/renoir_small.png
renoir_full: data/output/renoir.png

data/output/monet_tiny.png:
	$(PYTHON) src/generate.py --input-style-file=data/samples_tiny/monet.png \
							  --input-map-file=data/samples_tiny/monet_style.png \
							  --output-file=data/output/monet_tiny.png \
							  --output-map-file=data/samples_tiny/monet_out_style.png \
							  --num-phases=200 \
							  --map-channel-weight=5000 \
							  --plot-interval=$(PLOT_INTERVALS)

data/output/monet_small.png:
	$(PYTHON) src/generate.py --input-style-file=data/samples_small/monet.png \
							  --input-map-file=data/samples_small/monet_style.png \
							  --output-file=data/output/monet_small.png \
							  --output-map-file=data/samples_small/monet_out_style.png \
							  --num-phases=200 \
							  --map-channel-weight=5000 \
							  --plot-interval=$(PLOT_INTERVALS)

data/output/monet_full.png:
	$(PYTHON) src/generate.py --input-style-file=data/samples/monet.png \
							  --input-map-file=data/samples/monet_style.png \
							  --output-file=data/output/monet.png \
							  --output-map-file=data/samples/monet_out_style.png \
							  --num-phases=200 \
							  --map-channel-weight=5000 \
							  --plot-interval=$(PLOT_INTERVALS)

data/output/renoir_tiny.png:
	$(PYTHON) src/generate.py --input-style-file=data/samples_tiny/renoir.png \
							  --input-map-file=data/samples_tiny/renoir_style.png \
							  --output-file=data/output/renoir_tiny.png \
							  --output-map-file=data/samples_tiny/renoir_out_style.png \
							  --num-phases=200 \
							  --map-channel-weight=5000 \
							  --plot-interval=$(PLOT_INTERVALS)

data/output/renoir_small.png:
	$(PYTHON) src/generate.py --input-style-file=data/samples_small/renoir.png \
							  --input-map-file=data/samples_small/renoir_style.png \
							  --output-file=data/output/renoir_small.png \
							  --output-map-file=data/samples_small/renoir_out_style.png \
							  --num-phases=200 \
							  --map-channel-weight=5000 \
							  --plot-interval=$(PLOT_INTERVALS)

data/output/renoir_full.png:
	$(PYTHON) src/generate.py --input-style-file=data/samples/renoir.png \
							  --input-map-file=data/samples/renoir_style.png \
							  --output-file=data/output/renoir.png \
							  --output-map-file=data/samples/renoir_out_style.png \
							  --num-phases=200 \
							  --map-channel-weight=5000 \
							  --plot-interval=$(PLOT_INTERVALS)