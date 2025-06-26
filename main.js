const canvas=document.getElementById("carCanvas");
canvas.width=200;

const ctx = canvas.getContext("2d");
const road=new Road(canvas.width/2,canvas.width*0.9);

let train = true;

// const env = new CarRLEnvironment(-5000, "Keys");
// let obs = env.resetEnvironment();
// let state = obs.sensors;

let env, obs, state, gradSum, batchCount
const policy = new DQN([5,16,4]);
const target = policy.clone();

let epsilon = 0.2;
let stepCount = 0;
let total_reward = 0;
let batch_size = 32;
let num_episodes = 500;
let episode = 0;

function resetEnv(){
    env = new CarRLEnvironment(-5000, "Keys");
    obs = env.resetEnvironment();
    state = obs.sensors;
    gradSum = policy.makeEmptyGrad();
    stepCount = 0;
    total_reward = 0;

}
function animate(){
    if (train){
        if (!env.done && stepCount < 10000){
            console.log("env.done", env.done);
            console.log("obs.done", obs.done);


            let action = Math.random() < epsilon ? Math.floor(Math.random() * 4) : argmax(policy.forward(state));
            console.log("action", action);
                
            obs = env.step(action)
            state = obs.next_state.sensors;
            const grad = policy.compute_gradient(obs, target);
            gradSum = DQN.add_gradient(gradSum, grad);

            batchCount++;

            if (batchCount == batch_size || obs.done){
                let avg_grad = DQN.divideGrad(gradSum, batchCount);
                policy.updateParams(avg_grad, lr = 0.001);
                target.softUpdateFrom(policy);
                gradSum = policy.makeEmptyGrad();
                batchCount = 0;
            }
            env.render(ctx, canvas);


            total_reward += obs.reward;
            stepCount++;
        }

        env.render(ctx, canvas);


        if (env.done || stepCount >= 10_000){
            console.log(`Episode ${episode + 1}
                         reward = ${total_reward.toFixed(2)}`);

            epsilon = Math.max(0.01, epsilon * 0.95)

            episode++;
            if (episode < num_episodes){
                resetEnv();
            }
            else{
                train = false;
                console.log("Training complete");
                // Save the trained policy parameters
                const trainedParams = policy.getParameters();
                localStorage.setItem('trained_car_policy', JSON.stringify(trainedParams));
                console.log("Trained policy parameters saved to localStorage");
                
                // Optionally save additional training metadata
                const trainingMetadata = {
                    episodes: num_episodes,
                    finalReward: total_reward,
                    stepCount: stepCount,
                    timestamp: new Date().toISOString()
                };
                localStorage.setItem('training_metadata', JSON.stringify(trainingMetadata));
                console.log("Training metadata saved");
                return;
            }
            
            
        }

        requestAnimationFrame(animate);
    }
        

}

resetEnv();
requestAnimationFrame(animate);