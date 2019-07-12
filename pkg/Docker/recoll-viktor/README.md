
docker-recoll-webui
===================


**Recoll**
==========

it finds keywords inside documents as well as file names.
Project home : http://www.lesbonscomptes.com/recoll
<b>Recoll</b> is a great program written to index data on your computer, fileserver, remote-data. It uses the powerful <i>xapian-library</i> - Project home: https://xapian.org/

To check detailed features for <i>Recoll</i> check out : http://www.lesbonscomptes.com/recoll/features.html

This container will pull Debian Jessie, download Recoll source.list, then install all the needed dependencies. It will also pull the front end for this container, which is located at https://github.com/koniu/recoll-webui

<b>How to install</b>
1. Clone this project to your computer. 

   On *Linux* `git clone https://github.com/viktor-c/docker-recoll-webui`
2. Edit the file recoll.conf 

   For *huge amount* of info about what options are possible check out *The Manual* [https://www.lesbonscomptes.com/recoll/usermanual/webhelp/docs/RCL.INSTALL.CONFIG.RECOLLCONF.html].
2. Make this file executable
   On *Linux* `chmod u+x container.sh`
3. In terminal run `./container.sh build` to build the image, then start the created container. 
4. The start you container with `./container.sh start /path/to/your/local/collection/of/data` or `./container.sh start`, in the latter form it will pick up the default path from inside *container.sh* file
   Both commands will return the *container id* and the *IP-Address* where you can find recoll-webui
4. Access the docker-recoll-webui frontend by clicking the link or copying the IP-Adress
   

**Notes**
=========

1. Note on using the frontend: 
If you want to open a file in your browser after a successful query you might notice that the URL is incomplete. Check out the **settings** on the webpage from the container, here you can set the full path from your computer.
   
   The format you should use is `file:///path/to/your/local/collection/of/data`
2. Even if you set this path, opening files in your browser might not work. This is a safety issue in Firefox, Chrome, Opera. You cannot open local files from a remove website. To workaround this, please check out the recoll-webui Readme https://github.com/viktor-c/recoll-webui#opening-files-via-local-links
   Please note you have to give the IP-Address with wich the container starts. 
   If you still cannot open the files in Firefox / Opera, *double check the path*, see Note 1 above and* double check the IP-Address of the container*
3. On the first run of recoll in the collection folder it will also index the files. It can take more or less time, depending on where the files are located and how big the collection is.
