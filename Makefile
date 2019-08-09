NOWEBPATH	= /usr
WEAVE   	= $(NOWEBPATH)/bin/noweave
TANGLE    	= $(NOWEBPATH)/bin/notangle

all: python matlab doc
python: simceo.nw simceo.py
	mkdir -p calibration_dbs
pythonclient: 
	$(TANGLE) -Rsimceoclient.py simceo.nw > simceoclient.py
dosapi: 
	$(TANGLE) -Rdos.yaml simceo.nw > etc/dos/dos.yaml
	$(TANGLE) -Rdos.py simceo.nw > dos/dos.py
	$(TANGLE) -Rdriver.py simceo.nw > dos/driver.py
matlab: simceo.nw maskdoc
	mkdir -p +ceo
	$(TANGLE) -Rbroker.m simceo.nw > +ceo/broker.m
	$(TANGLE) -Rdealer.m simceo.nw > +ceo/dealer.m
	$(TANGLE) -Rliftprm.m simceo.nw > +ceo/liftprm.m
	$(TANGLE) -RSCEO.m simceo.nw > SCEO.m
maskdoc: simceo.nw
	mkdir -p masks
	$(TANGLE) -ROpticalPath.md simceo.nw > masks/OpticalPath.md
	$(TANGLE) -RGMTMirror.md simceo.nw > masks/GMTMirror.md
	make -C masks all
server: simceo.nw
	mkdir -p etc
	$(TANGLE) -RCEO.sh simceo.nw > etc/.CEO.sh
	make -C etc/ all

publish:
	aws s3 cp etc/startstop.js s3://gmto.modeling/
	aws s3 cp etc/simceo_aws_server.html s3://gmto.modeling/

doc: simceo.nw simceo.tex
	make -C doc/ all

zip: matlab maskdoc doc
	zip -r simceo.zip +ceo/* doc/simceo_refman.pdf etc/ec2runinst.json etc/cloudwatch.json etc/simceo.json  jsonlab/* matlab-zmq/* masks/*.html masks/*.css models/* CEO.slx SCEO.m slblocks.m simceo.nw

ipython:
	env LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONPATH=/home/ubuntu/CEO/python ipython

.SUFFIXES: .nw .tex .py .m

.nw.tex:
	$(WEAVE) -delay -index $< > doc/$@

.nw.py:
	$(TANGLE) -R$@ $< > $@

.nw.m:
	$(TANGLE) -R$@ $< > $@

