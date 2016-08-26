NOWEBPATH	= /usr
WEAVE   	= $(NOWEBPATH)/bin/noweave
TANGLE    	= $(NOWEBPATH)/bin/notangle

all: python matlab doc
python: simceo.nw simceo.py
	mkdir -p calibration_dbs
matlab: simceo.nw maskdoc
	mkdir -p +ceo
	$(TANGLE) -Rbroker.m simceo.nw > +ceo/broker.m
	$(TANGLE) -Rmessages.m simceo.nw > +ceo/messages.m
	$(TANGLE) -RSCEO.m simceo.nw > SCEO.m
maskdoc: simceo.nw
	mkdir -p masks
	$(TANGLE) -ROpticalPath.md simceo.nw > masks/OpticalPath.md
	$(TANGLE) -RGMTMirror.md simceo.nw > masks/GMTMirror.md
	make -C masks all
server: simceo.nw
	mkdir -p etc
	$(TANGLE) -RCEO.sh simceo.nw > etc/.CEO.sh
	$(TANGLE) -Rceo\\_server simceo.nw > etc/.ceo_server
	make -C etc/ all

doc: simceo.nw simceo.tex
	make -C doc/ all

zip: matlab maskdoc doc
	zip -r simceo.zip +ceo/* doc/simceo_refman.pdf etc/gmto.control.credentials.csv  jsonlab/* matlab-zmq/* masks/*.html masks/*.css models/* CEO.slx SCEO.m slblocks.m simceo.nw

ipython:
	env LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONPATH=/home/ubuntu/CEO/python ipython

.SUFFIXES: .nw .tex .py .m

.nw.tex:
	$(WEAVE) -delay -index $< > doc/$@

.nw.py:
	$(TANGLE) -R$@ $< > $@

.nw.m:
	$(TANGLE) -R$@ $< > $@

