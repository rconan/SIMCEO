NOWEBPATH	= /usr
WEAVE   	= $(NOWEBPATH)/bin/noweave
TANGLE    	= $(NOWEBPATH)/bin/notangle

all: python matlab maskdoc
python: simulink.nw simulink.py
matlab: simulink.nw
	mkdir -p +ceo
	$(TANGLE) -Rbroker.m simulink.nw > +ceo/broker.m
	$(TANGLE) -Rmessages.m simulink.nw > +ceo/messages.m
	$(TANGLE) -RSCEO.m simulink.nw > SCEO.m
maskdoc: simulink.nw
	mkdir -p masks
	$(TANGLE) -ROpticalPath.md simulink.nw > masks/OpticalPath.md
	$(TANGLE) -RGMTMirror.md simulink.nw > masks/GMTMirror.md
	make -C masks all
server: simulink.nw
	mkdir -p etc
	$(TANGLE) -RCEO.sh simulink.nw > etc/.CEO.sh
	$(TANGLE) -Rceo\\_server simulink.nw > etc/.ceo_server
	make -C etc/ all

doc: simulink.nw simulink.tex
	mv simulink.tex doc/simceo.tex
	make -C doc/ all

zip: matlab maskdoc doc
	zip -r simceo.zip +ceo/* CEO.slx doc/simceo_refman.pdf etc/gmto* jsonlab/* matlab-zmq/* masks/*.html masks/*.css optical_path.slx SCEO.m slblocks.m

ipython:
	env LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONPATH=/home/ubuntu/CEO/python ipython

.SUFFIXES: .nw .tex .py .m

.nw.tex:
	$(WEAVE) -delay -index $< > $@

.nw.py:
	$(TANGLE) -R$@ $< > $@

.nw.m:
	$(TANGLE) -R$@ $< > $@

