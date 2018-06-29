
Integration of leaflet into Jekyll using knits, markdown in R Studio

The R htmlwidget leaflet generates code to display javascript leaflet maps in html pages.

A few things need to be setup in the Jekyll site in order to get this to work with markdown and Jekyll static webpage generation.

- add relevant .js libraries to the /assets folder
	- (future work is to figure out web links for this?)
- add html script tags to the head.html file in the Jekyll pages _include folder

- make sure the .md post includes the correct tags


ISSUES:
- software updates may break this. The copied .js files will not update while code and functionality in R will.

- a .wrapper class css setting is present in the copied .js files and directories. This changes the Jekyll pages header wrapper.
	=> fix in Jekyll header css by renaming class