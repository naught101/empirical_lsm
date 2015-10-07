#!/usr/bin/env bash

pweave -s ipython -f pandoc model_search.mdw
pandoc --verbose -s model_search.md -o model_search.html -c pweave.css
rm -v model_search.md

