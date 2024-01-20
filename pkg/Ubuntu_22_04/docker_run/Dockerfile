#!/usr/bin/env -S bash -c "docker run -p 8080:8080 -it --rm \$(docker build --progress plain -f \$0 . 2>&1 | tee /dev/stderr | tac | grep -m 1 -oP 'sha256:[0-9a-f]*')"

# syntax = docker/dockerfile:1.4.0

FROM node:20

WORKDIR /root

RUN npm install sqlite3

RUN <<EOF cat >/root/schema.sql
  CREATE TABLE IF NOT EXISTS clicks (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    time INTEGER NOT NULL
  );
EOF

RUN <<EOF cat >/root/server.js
  const fs = require("fs");
  const http = require("http");
  const sqlite3 = require("sqlite3");

  const db = new sqlite3.Database(":memory:");
  db.run(fs.readFileSync("/root/schema.sql", "utf8"));

  const html = fs.readFileSync("/root/index.html", "utf8");
  const server = http.createServer((req, res) => {
    db.run("INSERT INTO clicks(time) VALUES(unixepoch())");

    const data = [];
    db.each(
      "SELECT time as t, COUNT(*) as n FROM clicks WHERE t > unixepoch()-4*60*60 GROUP BY t-t%60",
      (_, { t, n }) => data.push([Math.floor(t/60), n]),
      () => {
        res.writeHead(200, { "content-type": "text/html" });
        res.end(html.replace("__DATA__", JSON.stringify(data)));
      },
    );
  });

  server.listen(8080, "", () => console.log("serving :8080..."));
EOF

RUN <<EOF cat >/root/index.html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>#!/usr/bin/env docker run</title>
  </head>
  <body style="font-family: monospace; font-size; 12px; ">
    <div style="position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; background-size: 5vh 5vh; background-image: linear-gradient(to right, #f0f0f0 1px, transparent 1px), linear-gradient(to bottom, #f0f0f0 1px, transparent 1px); "></div>
    <span style="position: absolute; top: 1vh; left: 5vh;">Page loads over time (last 4 hours)</span>
    <span id="max" style="position: absolute; top: 5vh; left: 1vh;"></span>
    <span id="min" style="position: absolute; top: 95vh; left: 1vh;">0</span>
    <canvas id="canvas" style="position: absolute; top: 5vh; left: 5vw; "></canvas>
    <script>
      (() => {
        const el = document.getElementById("canvas"), ctx = el.getContext("2d");
        el.width = 0.9 * window.innerWidth * window.devicePixelRatio;
        el.height = 0.9 * window.innerHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        const data = __DATA__;
        const max = data.reduce((prev, [_, n]) => (n > prev ? n : prev), 0);
        document.getElementById("max").innerText = max;

        ctx.beginPath();
        ctx.moveTo(0, el.height);

        const draw = (t, n) => {
          const [x, y] = [el.width * (t-data[0][0])/240, el.height * (1 - n/max)];
          ctx.lineTo(x, y);
          ctx.moveTo(x, y);
        }

        let last = -1;
        for (const [t, n] of data) {
          if (last != -1 && t > last + 1) {
            draw(last + 0.1, 0);
            draw(t - 0.1, 0);
          }
          draw(t, n);
          last = t;
        }
        ctx.stroke();
      })();
    </script>
  </body>
</html>
EOF

CMD node /root/server.js
