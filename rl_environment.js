class CarRLEnvironment {
    constructor(finishLineY = -5000, controlType = "AI") {
        this.FINISH_LINE_Y = finishLineY;
        this.road = new Road(100, 180); // Centered road
        this.controlType = controlType;
        this.resetEnvironment();
        
        // RL Configuration
        this.ACTION_SPACE = 4; // [forward, left, right, reverse]
        this.STATE_SPACE = 5; // 5 sensor readings
        this.MAX_STEPS = 10000; // Maximum steps per episode
        this.currentStep = 0;
        
        // Reward parameters
        this.CRASH_PENALTY = -100;
        this.FINISH_REWARD = 10000;
        this.PROGRESS_REWARD_SCALE = 1.0;
        this.SPEED_REWARD_SCALE = 0.1;
        this.TIME_PENALTY = -0.5;
        
        this.lastY = 100; // Track progress
        
        // Create network only for AI control
        if (controlType === "AI") {
            this.network = new NeuralNetwork([this.STATE_SPACE, 6, this.ACTION_SPACE]);
        }
    }
    
    resetEnvironment() {
        // Create car with appropriate control type
        this.car = new Car(this.road.getLaneCenter(1), 100, 30, 50, this.controlType, 3, this.network);
        
        // Generate new traffic
        this.traffic = this.generateTraffic();
        
        // Reset tracking variables
        this.currentStep = 0;
        this.lastY = 100;
        this.done = false;
        
        return this.getState();
    }
    
    generateTraffic() {
        const traffic = [];
        const SEGMENT_LENGTH = 200;
        const CAR_SPACING = 150;
        const numberOfSegments = Math.floor(Math.abs(this.FINISH_LINE_Y + 3000) / SEGMENT_LENGTH);
        
        for (let segment = 0; segment < numberOfSegments; segment++) {
            const segmentStart = -100 - (segment * SEGMENT_LENGTH);
            const carsInSegment = Math.floor(Math.random() * 3); // 0, 1, or 2 cars
            
            if (carsInSegment > 0) {
                const availableLanes = [0, 1, 2];
                const occupiedLanes = [];
                
                for (let i = 0; i < carsInSegment; i++) {
                    const laneIndex = Math.floor(Math.random() * availableLanes.length);
                    const chosenLane = availableLanes.splice(laneIndex, 1)[0];
                    occupiedLanes.push(chosenLane);
                }
                
                occupiedLanes.forEach(lane => {
                    const y_pos = segmentStart - Math.random() * (SEGMENT_LENGTH - CAR_SPACING);
                    traffic.push(new Car(this.road.getLaneCenter(lane), y_pos, 30, 50, "Dummy", 1));
                });
            }
        }
        
        return traffic;
    }
    
    getState() {
        // Update sensor readings
        this.car.sensor.update(this.road.borders, this.traffic);
        
        // Normalize sensor readings to [0, 1]
        const sensorReadings = this.car.sensor.readings.map(reading => 
            reading === null ? 0 : 1 - reading.offset
        );
        
        // Additional state information (optional)
        const additionalState = {
            speed: this.car.speed / this.car.maxSpeed, // Normalized speed
            angle: this.car.angle / (2 * Math.PI), // Normalized angle
            distanceToFinish: Math.max(0, (this.car.y - this.FINISH_LINE_Y) / Math.abs(this.FINISH_LINE_Y)),
            lanePosition: (this.car.x - this.road.getLaneCenter(0)) / (this.road.getLaneCenter(2) - this.road.getLaneCenter(0))
        };
        
        return {
            sensors: sensorReadings,
            additional: additionalState,
            raw: {
                carY: this.car.y,
                carX: this.car.x,
                carSpeed: this.car.speed,
                carAngle: this.car.angle,
                damaged: this.car.damaged
            }
        };
    }
    
    step(action) {
        if (this.done) {
            // console.warn("Environment is done. Call reset() to start new episode.");
            return this.getStepResult(); // this needs to be changed as no action is being taken, replace with final result
        }
        
        const prevState = this.getState();
        
        // Apply action to car controls
        this.applyAction(action);
        
        // Update environment
        this.updateEnvironment();
        
        // Get new state
        const newState = this.getState();
        
        // Calculate reward
        const reward = this.calculateReward(prevState, action, newState);
        
        // Check if episode is done
        this.checkDone();
        
        this.currentStep++;
        
        return this.getStepResult(prevState, action, reward, newState);
    }
    
    applyAction(action) {
        
        // Apply action
        // Action can be either:
        // 1. Array of 4 values [forward, left, right, reverse] (0 or 1)
        // 2. Single integer (0=forward, 1=left, 2=right, 3=reverse)
        // 3. Array of 4 continuous values [0, 1] for each control
        console.log(action);
        if (Array.isArray(action)) {
            if (action.length === 4) {
                // Continuous or binary action array
                this.car.controls.forward = action[0] > 0.5;
                this.car.controls.left = action[1] > 0.5;
                this.car.controls.right = action[2] > 0.5;
                this.car.controls.reverse = action[3] > 0.5;
            }
        } else if (typeof action === 'number') {
            // Discrete action
            switch(action) {
                case 0: this.car.controls.forward = true; break;
                case 1: this.car.controls.left = true; break;
                case 2: this.car.controls.right = true; break;
                case 3: this.car.controls.reverse = true; break;
            }
        }
    }
    
    updateEnvironment() {
        // Update traffic
        for (let i = 0; i < this.traffic.length; i++) {
            this.traffic[i].update(this.road.borders, []);
        }
        
        // Update car
        this.car.update(this.road.borders, this.traffic);
    }
    
    calculateReward(prevState, action, newState) {
        let reward = 0;
        
        // Crash penalty
        if (this.car.damaged) {
            reward += this.CRASH_PENALTY;
            return reward; // End episode immediately
        }
        
        // Finish reward
        if (this.car.y <= this.FINISH_LINE_Y) {
            reward += this.FINISH_REWARD;
            // Bonus for finishing faster
            const timeBonus = Math.max(0, (this.MAX_STEPS - this.currentStep) / this.MAX_STEPS * 100);
            reward += timeBonus;
            return reward;
        }
        
        // Progress reward (moving forward)
        const progress = this.lastY - this.car.y; // Positive when moving forward
        if (progress > 0) {
            reward += progress * this.PROGRESS_REWARD_SCALE;
        } else {
            // Small penalty for going backwards or not moving
            reward += progress * 0.5;
        }
        this.lastY = this.car.y;
        
        // Speed reward (encourage forward movement)
        if (this.car.speed > 0) {
            reward += this.car.speed * this.SPEED_REWARD_SCALE;
        }
        
        // Time penalty (encourage efficiency)
        reward += this.TIME_PENALTY;
        
        // Lane keeping bonus (stay in lanes, avoid erratic movement)
        const laneCenter1 = this.road.getLaneCenter(1);
        const distanceFromCenter = Math.abs(this.car.x - laneCenter1);
        if (distanceFromCenter < 30) { // Within lane
            reward += 0.5;
        }
        
        return reward;
    }
    
    checkDone() {
        this.done = this.car.damaged || 
                   this.car.y <= this.FINISH_LINE_Y || 
                   this.currentStep >= this.MAX_STEPS;
    }
    
    getStepResult(prevState = null, action = null, reward = 0, newState = null) {
        return {
            state: prevState,
            action: action,
            reward: reward,
            next_state: newState || this.getState(),
            done: this.done,
            info: {
                step: this.currentStep,
                car_y: this.car.y,
                car_x: this.car.x,
                speed: this.car.speed,
                damaged: this.car.damaged,
                finished: this.car.y <= this.FINISH_LINE_Y,
                distance_to_finish: Math.max(0, this.car.y - this.FINISH_LINE_Y)
            }
        };
    }
    
    // Utility methods for RL training
    getActionSpace() {
        return {
            type: "discrete", // or "continuous"
            size: this.ACTION_SPACE,
            description: ["forward", "left", "right", "reverse"]
        };
    }
    
    getStateSpace() {
        return {
            sensors: {
                size: this.STATE_SPACE,
                range: [0, 1],
                description: "5 sensor readings (normalized)"
            },
            additional: {
                speed: [0, 1],
                angle: [0, 1],
                distanceToFinish: [0, 1],
                lanePosition: [0, 1]
            }
        };
    }
    
    // Render method for visualization (optional)
    render(ctx, canvas) {
        if (!ctx || !canvas) return;
        
        canvas.height = window.innerHeight;
        
        ctx.save();
        ctx.translate(0, -this.car.y + canvas.height * 0.7);
        
        this.road.draw(ctx);
        
        // Draw finish line
        this.drawFinishLine(ctx, this.FINISH_LINE_Y);
        
        // Draw traffic
        for (let i = 0; i < this.traffic.length; i++) {
            this.traffic[i].draw(ctx, "red");
        }
        
        // Draw car
        let carColor = this.car.damaged ? "gray" : 
                      this.car.y <= this.FINISH_LINE_Y ? "gold" : "blue";
        this.car.draw(ctx, carColor, true);
        
        ctx.restore();
        
        // Draw info
        this.drawInfo(ctx);
    }
    
    drawFinishLine(ctx, y) {
        const roadLeft = this.road.borders[0][0].x;
        const roadRight = this.road.borders[1][0].x;
        const stripeWidth = 20;
        
        ctx.fillStyle = "white";
        ctx.fillRect(roadLeft, y - 10, roadRight - roadLeft, 20);
        
        ctx.fillStyle = "black";
        for(let x = roadLeft; x < roadRight; x += stripeWidth) {
            for(let i = 0; i < 2; i++) {
                const offsetX = (i % 2) * stripeWidth / 2;
                if((x + offsetX - roadLeft) / stripeWidth % 2 < 1) {
                    ctx.fillRect(x + offsetX, y - 10 + i * 10, stripeWidth / 2, 10);
                }
            }
        }
        
        ctx.fillStyle = "yellow";
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "center";
        ctx.fillText("FINISH", (roadLeft + roadRight) / 2, y + 5);
    }
    
    drawInfo(ctx) {
        const state = this.getState();
        ctx.fillStyle = "white";
        ctx.font = "14px Arial";
        ctx.textAlign = "left";
        
        ctx.fillText(`Step: ${this.currentStep}/${this.MAX_STEPS}`, 10, 20);
        ctx.fillText(`Distance: ${Math.round(Math.max(0, this.car.y - this.FINISH_LINE_Y))}`, 10, 40);
        ctx.fillText(`Speed: ${this.car.speed.toFixed(2)}`, 10, 60);
        ctx.fillText(`Sensors: [${state.sensors.map(s => s.toFixed(2)).join(', ')}]`, 10, 80);
        
        if (this.done) {
            ctx.fillStyle = this.car.damaged ? "red" : "gold";
            ctx.font = "bold 16px Arial";
            ctx.fillText(this.car.damaged ? "CRASHED!" : "FINISHED!", 10, 110);
        }
    }
}

// Make it globally available
window.CarRLEnvironment = CarRLEnvironment; 