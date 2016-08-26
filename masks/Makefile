all:
	pandoc OpticalPath.md -c style.css -s -o OpticalPath.html
	pandoc GMTMirror.md -c style.css -s -o GMTMirror.html
	sed -i -e 's/4070a0/e9967a/g' GMTMirror.html
	sed -i -e 's/4070a0/e9967a/g' OpticalPath.html
