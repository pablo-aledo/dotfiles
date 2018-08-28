QuickStart
Using Docker to run Debian Lenny
Make a Dockerfile like this:
FROM debian/eol:lenny

RUN apt-get update

RUN apt-get install -y complearn-tools libcomplearn-dev qsearch-tools libqsearch-dev zlib1g-dev
Build a tagged image for complearn
docker build -t complearn .
Enter the image to use complearn and transfer files with docker cp or -v host mount docker options
docker run -i -t complearn bash
Inside the container shell you may run ncd or maketree. You may also use
docker run complearn ncd
and similar commands to run the tools from the host.
Command Names
ncd - computes the Normalized Compression Distance
maketree - generates a best-fitting binary tree from a given distance matrix.
Computing NCD: Default Settings
ncd, by default, uses the bzlib compressor and file input format. Two filenames are passed in as command-line arguments. The contents of the files are compressed and the NCD between two files is returned.

Example:
$ ncd filename1 filename2
Selecting a Compressor
There are currently many compressors supported by the ncd command-line tool: bzlib, zlib and blocksort for example. A compressor may be selected by adding a -c or --compressor option, followed by the compressor type.

Option:
-c, --compressor=[ bzlib | zlib | blocksort ]

Examples:
$ ncd -c zlib filename1 filename2
$ ncd --compressor=blocksort filename1 filename2
Selecting an Input Mode
The input mode selected determines how a DataBlock Enumeration is created. The default mode is file mode and may be changed by adding command-line options which switch to a new mode. Such a command-line option is followed by one or more arguments, depending on the mode selected.

File Mode - Takes as an argument a filename whose contents are to be compressed.
String Literal Mode - Takes as an argument a string whose contents are to be compressed. By default, each string literal is separated by whitespace. For string literals containing white space, surround with double quotes.
Plain List Mode - Takes as an argument a filename which contains list of filenames to be individually compressed. Each filename is separated by a linebreak.
Term List Mode - Takes as an argument a filename whose contents contain a list of string literals to be individually compressed. Each string literal is separated by a linebreak.
Directory Mode - Takes as an argument the name of a directory whose file contents are individually compressed.
Options:
-f, --file-mode=FILE
-l, --literal-mode=STRING
-p, --plainlist-mode=FILE
-t, --termlist-mode=FILE
-d, --directory-mode=DIR
Examples:
$ ncd filename1 -l string1
computes the NCD between contents of a file and a string literal

$ ncd -l string1 -f filename1
computes the NCD between a string literal and the contents of a file

$ ncd -l string1 "s t r i n g 2"
computes the NCD between two string literals

$ ncd -p filename1 -f filename2
computes a list of NCDs for files in a plain list and a single file

$ ncd -t filename1 -d directory1
computes a matrix of NCDs for string literals in a term list and the files found in a directory

Creating an Unrooted Binary Tree: Default Settings
maketree, by default, takes a square distance matrix and computes a best-fitting unrooted binary tree. The results are put into a file called treefile.dot, which can then be used to create a layout using GraphViz's dot or neato.

The distance matrix should have been created using the ncd command, with the -b option. By default, the resulting distance matrix file is called distmatrix.clb, but the file name may be changed using the -o option.

There are two requirements of a distance matrix in order for maketree to work properly:

must be a square matrix - that is, the 1st and 2nd input arguments to the ncd command must be the same
dimensions must be 4x4 or greater
Examples:
$ ncd -b -t filename1 filename1
$ maketree distmatrix.clb
ncd creates a square distance matrix from a term list and saves the results in a file called distmatrix.clb. maketree stores a best-fitting unrooted binary tree in treefile.dot

$ ncd -b -o mydistmatrix.clb -t directory1 directory1
$ maketree mydistmatrix.clb
ncd creates a square distance matrix from the files in a directory and saves the results in a file called mydistmatrix.clb. maketree stores a best-fitting unrooted binary tree in treefile.dot

$ ncd -b -c zlib -p filename1 filename1
$ maketree distmatrix.clb
ncd creates a square distance matrix from the files in a plain list using the zlib compressor and saves the results in a file called distmatrix.clb. maketree stores a best-fitting unrooted binary tree in treefile.dot

Example:
$ maketree distmatrix.clb
Laying Out Your Tree:
You may use the neato command to create a postscript file showing your tree.

Example:
$ neato -Tps -Gsize=7,7 treefile.dot >tree.ps
neato creates a file tree.ps that depicts the generated tree in treefile.dot using a 7 by 7 inch drawing area and outputting postscript format. ]
