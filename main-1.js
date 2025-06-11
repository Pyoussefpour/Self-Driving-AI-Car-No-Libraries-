const canvas=document.getElementById("carCanvas");
canvas.width=200;

const ctx = canvas.getContext("2d");
const road=new Road(canvas.width/2,canvas.width*0.9);

// Race configuration
const FINISH_LINE_Y = -5000; // Finish line position
let raceFinished = false;
let raceStartTime = Date.now();
let finishTime = 0;

const car=new Car(road.getLaneCenter(1),100,30,50,"Keys");

// Smart traffic generation - prevents overlaps and road blocking
function generateTraffic() {
    const traffic = [];
    const SEGMENT_LENGTH = 200; // Length of each road segment
    const CAR_SPACING = 150; // Minimum distance between cars
    const numberOfSegments = Math.floor(Math.abs(FINISH_LINE_Y+3000) / SEGMENT_LENGTH);
    
    for (let segment = 0; segment < numberOfSegments; segment++) {
        const segmentStart = -100 - (segment * SEGMENT_LENGTH);
        // const segmentEnd = segmentStart - SEGMENT_LENGTH;
        
        // Randomly decide how many cars in this segment (0-2, never block all 3 lanes)
        const carsInSegment = Math.floor(Math.random() * 3); // 0, 1, or 2 cars
        
        if (carsInSegment > 0) {
            // Choose which lanes to place cars in (ensure at least one lane is always free)
            const availableLanes = [0, 1, 2];
            const occupiedLanes = [];
            
            for (let i = 0; i < carsInSegment; i++) {
                // Remove a random lane from available lanes
                const laneIndex = Math.floor(Math.random() * availableLanes.length);
                const chosenLane = availableLanes.splice(laneIndex, 1)[0];
                occupiedLanes.push(chosenLane);
            }
            
            // Place cars in chosen lanes
            occupiedLanes.forEach(lane => {
                const y_pos = segmentStart - Math.random() * (SEGMENT_LENGTH - CAR_SPACING);
                traffic.push(new Car(road.getLaneCenter(lane), y_pos, 30, 50, "Dummy", 2));
            });
        }
    }
    
    
    return traffic;
}
// const traffic = [ 
//     new Car(road.getLaneCenter(1), -150, 30, 50, "Dummy", 2),
//     new Car(road.getLaneCenter(0), -300, 30, 50, "Dummy", 2),
//     new Car(road.getLaneCenter(2), -300, 30, 50, "Dummy", 2),
//     new Car(road.getLaneCenter(1), -500, 30, 50, "Dummy", 2),
//     new Car(road.getLaneCenter(0), -700, 30, 50, "Dummy", 2),
// ]
const traffic = generateTraffic();

animate();

// Reset race function
function resetRace(){
    raceFinished = false;
    raceStartTime = Date.now();
    finishTime = 0;
    car.y = 100;
    car.damaged = false;
    car.speed = 0;
    car.angle = 0;
    car.x = road.getLaneCenter(1);
    
    // Regenerate traffic for new challenge
    traffic.length = 0; // Clear existing traffic
    const newTraffic = generateTraffic();
    traffic.push(...newTraffic);
    
}

// Draw finish line
function drawFinishLine(ctx, y) {
    const roadLeft = road.borders[0][0].x;
    const roadRight = road.borders[1][0].x;
    const stripeWidth = 20;
    
    // Draw checkered pattern
    ctx.fillStyle = "white";
    ctx.fillRect(roadLeft, y - 10, roadRight - roadLeft, 20);
    
    // Draw black squares for checkered pattern
    ctx.fillStyle = "black";
    for(let x = roadLeft; x < roadRight; x += stripeWidth) {
        for(let i = 0; i < 2; i++) {
            const offsetX = (i % 2) * stripeWidth / 2;
            if((x + offsetX - roadLeft) / stripeWidth % 2 < 1) {
                ctx.fillRect(x + offsetX, y - 10 + i * 10, stripeWidth / 2, 10);
            }
        }
    }
    
    // Draw "FINISH" text
    ctx.fillStyle = "yellow";
    ctx.font = "bold 12px Arial";
    ctx.textAlign = "center";
    ctx.fillText("FINISH", (roadLeft + roadRight) / 2, y + 5);
}

// Check for race finish
function checkRaceFinish() {
    if (!raceFinished && car.y <= FINISH_LINE_Y && !car.damaged) {
        raceFinished = true;
    }
}


function animate(){
   
    
    // Update car only if not finished and not damaged
    if (!raceFinished && !car.damaged) {
         // Update traffic
        for(let i=0;i<traffic.length;i++){
            traffic[i].update(road.borders,[]);
        }
        car.update(road.borders,traffic);
    }
    
    // Check for race finish
    checkRaceFinish();

    canvas.height=window.innerHeight;

    ctx.save();
    ctx.translate(0,-car.y+canvas.height*0.7);

    road.draw(ctx);
    
    // Draw finish line
    drawFinishLine(ctx, FINISH_LINE_Y);
    
    // Draw traffic
    for(let i=0;i<traffic.length;i++){
        traffic[i].draw(ctx,"red");
    }
    car.draw(ctx,"blue");

    ctx.restore();
    
    
    requestAnimationFrame(animate);
}