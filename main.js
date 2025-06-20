const carCanvas = document.getElementById("carCanvas");
carCanvas.width = 200;
const networkCanvas = document.getElementById("networkCanvas");
networkCanvas.width = 500;

const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");
const road = new Road(carCanvas.width / 2, carCanvas.width *0.9);
N = 100;
const cars  =generateCars(N);
let bestCar = cars[0];
if(localStorage.getItem("bestBrain")){
    bestCar.brain = JSON.parse(localStorage.getItem("bestBrain"));
}
// car.draw(carCtx);
const traffic = [ 
    new Car(road.getLaneCenter(1), -150, 30, 50, "Dummy", 2),
    new Car(road.getLaneCenter(0), -300, 30, 50, "Dummy", 2),
    new Car(road.getLaneCenter(2), -300, 30, 50, "Dummy", 2),
]


animate();

function save(){
    localStorage.setItem("bestBrain", JSON.stringify(bestCar.brain));
}

function discard(){
    localStorage.removeItem("bestBrain");
}

function generateCars(N=50){
    const cars = [];
    for (let i=0; i<N; i++){
        cars.push(new Car(road.getLaneCenter(1), -100, 30, 50, "AI",2));
    }
    return cars;
}

function animate(time){
    for (let i=0; i<traffic.length; i++){
        traffic[i].update(road.borders, []);
    }
    for (let i=0; i<cars.length; i++){
        cars[i].update(road.borders, traffic);
    }

    bestCar = cars.find(c => c.y == Math.min(...cars.map(c => c.y)));

    carCanvas.height = window.innerHeight;
    networkCanvas.height = window.innerHeight;

    carCtx.save();
    carCtx.translate(0, -bestCar.y + carCanvas.height * 0.7); // this is to move the car to the center of the canvas
    road.draw(carCtx);
    for (let i=0; i<traffic.length; i++){
        traffic[i].draw(carCtx, "red");
    }
    carCtx.globalAlpha = 0.2;
    for (let i=0; i<cars.length; i++){
        cars[i].draw(carCtx, "blue");
    }
    carCtx.globalAlpha = 1;
    bestCar.draw(carCtx, "blue", true);

    carCtx.globalAlpha = 0.2;
    carCtx.restore();
    networkCtx.lineDashOffset = - time / 50;
    Visualizer.drawNetwork(networkCtx, bestCar.brain);
    requestAnimationFrame(animate);  // this is a function that will call animate again and again
}