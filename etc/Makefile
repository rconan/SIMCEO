all:	
	sudo pkill -F /var/run/ceo_server.pid	
	sudo cp .CEO.sh CEO.sh
	sudo chown root:root CEO.sh
	sudo chmod u+x CEO.sh
	sudo cp CEO.sh /etc/init.d/
	sudo /etc/init.d/CEO.sh start
