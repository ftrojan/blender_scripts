const canvas = document.getElementById('canvas1');
const ctx = canvas.getContext('2d');
canvas.width = 1024;
canvas.height = 1024;

const keys = [];
let mouse = {
  x: 0,
  y: 0
}

const background = new Image();
background.src = "../pipeline_data/mustek001/texture_streets-v11.png";

canvas.addEventListener("mousemove", function(e){
  mouse.x = e.offsetX;
  mouse.y = e.offsetY;
})

let frameTime, now, frameStarted, drawTime;

function startAnimating(fps) {
  frameTime = 1000/fps;
  frameStarted = Date.now();
  animate();
}

function drawCursor(cursor) {
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, cursor.y);
  ctx.lineTo(canvas.width, cursor.y);
  ctx.moveTo(cursor.x, 0);
  ctx.lineTo(cursor.x, canvas.height);
  ctx.stroke();
  ctx.font = '12px menlo';
  let text = cursor.x.toFixed(2) + "," + cursor.y.toFixed(2);
  ctx.fillText(text, cursor.x + 5, cursor.y - 5);
}

function drawAll(){
  ctx.drawImage(background, 0, 0, canvas.width, canvas.height);
  drawCursor(mouse);
}

function animate(){
  requestAnimationFrame(animate); // recursion
  now = Date.now();
  drawTime = now - frameStarted;
  if (drawTime > frameTime) {
    frameStarted = now - (drawTime % frameTime);
    drawAll();
  }
}

d3.json("../pipeline_data/mustek001/elevation_d3.json", function(d) {
  let el = d;
  console.log(el);
  const k = canvas.width / d.width;
  const delta = 10;
  const thick = 5;
  let emin = Math.floor(d3.min(el.values)/delta/thick)*delta*thick;
  let emax = Math.ceil(d3.max(el.values)/delta/thick)*delta*thick;
  let thresholds = d3.range(emin, emax, delta);
  console.log(thresholds);
  let color = d3.scaleSequentialLog(d3.extent(thresholds), d3.interpolateMagma);
  let x = d3.scaleLinear([d.x1, d.x2], [0, canvas.width]);
  let y = d3.scaleLinear([d.y1, d.y2], [canvas.height, 0]);
  let transform = ({type, value, coordinates}) => {
      return {type, value, coordinates: coordinates.map(rings => {
        return rings.map(points => {
          return points.map(([x, y]) => ([
            k * x,
            k * y
          ]));
        });
      })};
    }
  let contours = d3.contours()
    .size([d.width, d.height])
    .thresholds(thresholds)
    (d.values)
    .map(transform);
  console.log(contours);
  startAnimating(10);
});