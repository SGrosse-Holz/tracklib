SPHINXDIR = doc/sphinx
SPHINXBUILD = doc/sphinx/build
SPHINXSOURCE = doc/sphinx/source
TESTDIR = tests
TESTFILE = test.py
# TESTFILE = test_bild.py ###### CHANGE BACK TO test.py, this is just for dev
COVERAGEREPFLAGS = 
# COVERAGEREPFLAGS = --include=tracklib/analysis/bild/*
COVERAGEREPDIR = doc/coverage
DEVGUIDEDIR = doc/dev_guide
DEVGUIDEPDF = guide.pdf
MODULE = tracklib

.PHONY : doc tests devguide all clean mydoc mytests mydevguide myall myclean

all : doc devguide tests

doc :
	sphinx-apidoc -f -o $(SPHINXSOURCE) $(MODULE)
	@rm $(SPHINXSOURCE)/modules.rst
	@cd $(SPHINXSOURCE) && vim -n -S post-apidoc.vim
	cd $(SPHINXDIR) && $(MAKE) html

tests :
	cd $(TESTDIR) && coverage run $(TESTFILE)
	@mv $(TESTDIR)/.coverage .
	coverage html -d $(COVERAGEREPDIR) $(COVERAGEREPFLAGS)

devguide :
	cd $(DEVGUIDEDIR) && pandoc dev_guide.md -o $(DEVGUIDEPDF)

clean :
	-rm -r $(SPHINXBUILD)/*
	-rm -r $(COVERAGEREPDIR)/*
	-rm $(DEVGUIDEDIR)/$(DEVGUIDEPDF)
	-rm .coverage

# Personal convenience targets
DUMPPATH = "/home/simongh/Dropbox (MIT)/htmldump"
mydoc : doc
	cp -r $(SPHINXBUILD)/* $(DUMPPATH)/sphinx

mytests : tests
	cp -r $(COVERAGEREPDIR)/* $(DUMPPATH)/coverage

mydevguide : devguide
	cp -r $(DEVGUIDEDIR)/* $(DUMPPATH)/dev_guide

myall : mydoc mydevguide mytests

myclean : clean
	-rm -r $(DUMPPATH)/sphinx/*
	-rm -r $(DUMPPATH)/coverage/*
	-rm -r $(DUMPPATH)/dev_guide/*
