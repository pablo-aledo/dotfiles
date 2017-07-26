curl --data "&Name=Pablo%20Aledo&JobTitle=Phd&Company=UC&Street=Av.Castros&City=Santander&Country=Spain&Postcode=3905&Email=pablo.aledo@gmail.com&Homepage=www.unican.es&Telephone=942&Fax=942&Agreement=on" http://www.it.uu.se/research/group/darts/uppaal/download/form.php -o /tmp/uppaal

sessid=`cat /tmp/uppaal | grep PHPSESSID | cut -d"=" -f4 | cut -d"'" -f1`

curl --data "Name=Pablo+Aledo&JobTitle=Phd&Company=UC&Street=Avda.Castros&City=Santander&Country=Spain&Postcode=39005&Email=pablo.aledo%40gmail.com&Homepage=www.unican.es&Telephone=942&Fax=942&Agreement=on&id=0&subid=4" --cookie "PHPSESSID=$sessid" "http://www.it.uu.se/research/group/darts/uppaal/download/form.php?"  -o /tmp/uppaal.zip

curl --data "Name=Pablo+Aledo&JobTitle=Phd&Company=UC&Street=Avda.Castros&City=Santander&Country=Spain&Postcode=39005&Email=pablo.aledo%40gmail.com&Homepage=www.unican.es&Telephone=942&Fax=942&Agreement=on&id=0&subid=4" --cookie "PHPSESSID=$sessid" "http://www.it.uu.se/research/group/darts/uppaal/download/form.php?"  -o /tmp/uppaal.zip

cd /
sudo unzip /tmp/uppaal.zip 
cd uppaal64-4.1.19

sudo ln -s $PWD/uppaal /bin/
sudo ln -s $PWD/bin-Linux/verifyta /bin/


