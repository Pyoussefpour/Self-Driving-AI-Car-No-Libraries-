// RL Environment Usage Example
// This demonstrates how to use the CarRLEnvironment for reinforcement learning

class RLExample {
    constructor() {
        this.env = null;
        this.episode = 0;
        this.totalReward = 0;
        this.stepCount = 0;
    }
    
    // Initialize the environment
    initializeEnvironment(controlType = "AI") {
        this.env = new CarRLEnvironment(-5000, "Keys"); // Finish line at -5000 
        console.log("🤖 RL Environment initialized!");
        console.log("Action Space:", this.env.getActionSpace());
        console.log("State Space:", this.env.getStateSpace());
        return this.env;
    }
    
    // Run a single episode with random actions (for testing)
    runRandomEpisode() {
        if (!this.env) {
            this.initializeEnvironment();
        }
        
        // Reset environment
        let state = this.env.resetEnvironment();
        this.episode++;
        this.totalReward = 0;
        this.stepCount = 0;
        
        console.log(`\n🏁 Starting Episode ${this.episode}`);
        console.log("Initial State:", state);
        
        // Run episode
        while (!this.env.done && this.stepCount < 1000) {
            // Random action selection
            const action = Math.floor(Math.random() * 4); // 0-3
            
            // Take step
            const result = this.env.step(action);
            
            this.totalReward += result.reward;
            this.stepCount++;
            
            
            if (result.done) {
                console.log(`\n🏆 Episode ${this.episode} finished!`);
                console.log(`Steps: ${this.stepCount}`);
                console.log(`Total Reward: ${this.totalReward.toFixed(2)}`);
                console.log(`Reason: ${result.info.damaged ? 'Crashed' : result.info.finished ? 'Finished' : 'Max steps'}`);
                break;
            }
        }
        
        return {
            episode: this.episode,
            steps: this.stepCount,
            totalReward: this.totalReward,
            finished: this.env.car.y <= this.env.FINISH_LINE_Y,
            crashed: this.env.car.damaged
        };
    }
    
    // Run multiple episodes for testing
    runMultipleEpisodes(numEpisodes = 5) {
        const results = [];
        
        for (let i = 0; i < numEpisodes; i++) {
            const result = this.runRandomEpisode();
            results.push(result);
        }
        
        // Summary statistics
        const avgReward = results.reduce((sum, r) => sum + r.totalReward, 0) / results.length;
        const avgSteps = results.reduce((sum, r) => sum + r.steps, 0) / results.length;
        const finishRate = results.filter(r => r.finished).length / results.length;
        const crashRate = results.filter(r => r.crashed).length / results.length;
        
        console.log(`\n📊 Summary of ${numEpisodes} episodes:`);
        console.log(`Average Reward: ${avgReward.toFixed(2)}`);
        console.log(`Average Steps: ${avgSteps.toFixed(1)}`);
        console.log(`Finish Rate: ${(finishRate * 100).toFixed(1)}%`);
        console.log(`Crash Rate: ${(crashRate * 100).toFixed(1)}%`);
        
        return results;
    }
    
    // Demonstrate different action types
    demonstrateActions() {
        if (!this.env) {
            this.initializeEnvironment();
        }
        
        this.env.resetEnvironment();
        
        console.log("\n🎮 Demonstrating different action types:");
        
        // Discrete action
        console.log("1. Discrete action (forward):");
        let result = this.env.step(0);
        console.log("Result:", result);
        
        // Binary array action
        console.log("\n2. Binary array action [forward, left, right, reverse]:");
        result = this.env.step([1, 0, 0, 0]); // Forward only
        console.log("Result:", result);
        
        // Continuous array action
        console.log("\n3. Continuous array action:");
        result = this.env.step([0.8, 0.3, 0.0, 0.0]); // Mostly forward, slight left
        console.log("Result:", result);
    }
    
    // Visualize an episode (if canvas is available)
    visualizeEpisode(canvas, ctx) {
        if (!this.env) {
            this.initializeEnvironment();
        }
        
        this.env.resetEnvironment();
        
        const animate = () => {
            if (!this.env.done) {
                // Random action for demo
                const action = Math.floor(Math.random() * 4);
                const result = this.env.step(action);
                
                // Render
                this.env.render(ctx, canvas);
                
                // Continue animation
                setTimeout(() => requestAnimationFrame(animate), 50); // Slower for visibility
            } else {
                console.log("Episode finished in visualization");
            }
        };
        
        animate();
    }
}

// Make it globally available
window.RLExample = RLExample;

// Auto-run example when loaded
window.addEventListener('load', () => {
    console.log("🚀 RL Environment Example loaded!");
    console.log("Available commands:");
    console.log("- const rl = new RLExample()");
    console.log("- rl.initializeEnvironment()");
    console.log("- rl.runRandomEpisode()");
    console.log("- rl.runMultipleEpisodes(10)");
    console.log("- rl.demonstrateActions()");
    console.log("- rl.visualizeEpisode(canvas, ctx) // if you have canvas");
    
    // Quick demo
    const rl = new RLExample();
    rl.demonstrateActions();
}); 