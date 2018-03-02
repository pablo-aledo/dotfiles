from canto.extra import *
import os

# Handlers when in Linux console or xterm
if os.getenv("TERM") == "linux":
    link_handler("echo \"%u\" >> ~/canto_links")
    image_handler("fbi \"%u\"", text=True, fetch=True)
else:
    link_handler("google-chrome \"%u\"")
    image_handler("feh \"%u\"", fetch=True)

# Max column width of 65 characters
#def resize_hook(cfg):
    #cfg.columns = cfg.width / 65

# Never discard items I haven't seen
never_discard("unread")

# I prefer change_tag to interval
# Uncomment these to use it too

# triggers.remove("interval")
# triggers.append("change_tag")

keys['/'] = search_filter
keys['y'] = yank

# Use [ / ] to switch between global filters
filters=[show_unread, None]

# Make unread items float to the top, when not
# using show_unread filter
default_tag_sorts([by_unread])

# Selected Feeds
add("http://hackaday.com/feed")
add("http://www.blackhat.com/BlackHatRSS.xml")
add("http://makezine.com/feed")
add("https://feeds.feedburner.com/Blendernation")
add("http://muylinux.com/feed")
add("https://jeremykun.com/feed")
add("http://blog.regehr.org/feed")
add("http://www.embedds.com/feed")
add("http://www.3dtotal.com/rss")
add("http://dangerousprototypes.com/blog/rss")
add("http://barrapunto.com/barrapunto.rss")
add("http://rss.slashdot.org/slashdot/Slashdot")
add("http://reddit.com/.rss")
add("https://ted.com/talks/rss")
add("http://thehackernews.com/feeds/posts/default")
add("http://blog.kaggle.com/feed")
add("http://karpathy.github.io/feed.xml")
add("http://nlpers.blogspot.com/feeds/posts/default")
add("https://jack-clark.net/feed/")
add("http://andrewgelman.com/feed/")
add("http://fastml.com/atom.xml")
add("http://flowingdata.com/feed")
add("http://www.becomingadatascientist.com/feed/")
add("http://www.datatau.com/rss")
add("https://www.r-bloggers.com/feed/")
add("https://news.ycombinator.com/rss")
add("http://www.codeproject.com/WebServices/ArticleRSS.aspx")
add("http://www.hackster.io/projects.rss?sort=recent")
add("http://deeplearning.net/feed/")
add("http://avxhm.se/ebooks/programming_development/rss.xml")
add("http://avxhm.se/ebooks/graphics_drawing_design/rss.xml")
add("http://avxhm.se/ebooks/engeneering_technology/rss.xml")
add("http://avxhm.se/ebooks/hardware/rss.xml")
add("http://avxhm.se/ebooks/science_books/rss.xml")
add("http://avxhm.se/ebooks/security_info/rss.xml")
add("http://avxhm.se/ebooks/software/rss.xml")
add("http://feeds.bbci.co.uk/news/rss.xml")
add("http://feeds.bbci.co.uk/news/technology/rss.xml")

# Some examples
# Uncomment if you've downloaded the script
# add("script:slashdotpolls -external")
#
# Simple password example
# add("http://feedparser.org/docs/examples/digest_auth.xml", username="test",
#        password="digest")
