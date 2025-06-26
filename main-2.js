const canvas=document.getElementById("carCanvas");
canvas.width=200;

const ctx = canvas.getContext("2d");
const road=new Road(canvas.width/2,canvas.width*0.9);

let train = true;

// const env = new CarRLEnvironment(-5000, "Keys");
// let obs = env.resetEnvironment();
// let state = obs.sensors;
const policy = new DQN([5,16,4]);
const target = policy.clone();

let epsilon = 0.1;
let stepCount = 0;
let total_reward = 0;
let batch_size = 32;
let num_episodes = 100;

function animate(env, obs, state){
    if (train){
        if (!env.done && stepCount < 10000){
            let gradSum = policy.makeEmptyGrad();
            let batchCount = 0
            while (batchCount < batch_size && !env.done){
                let action = Math.random() < epsilon ? Math.floor(Math.random() * 4) : argmax(policy.forward(state));
                obs = env.step(action)
                state = obs.state.sensors;

                const grad = policy.compute_gradient(obs, target);
                gradSum = DQN.add_gradient(gradSum, grad);
                
                env.render(ctx, canvas);

                requestAnimationFrame(() => animate(env, obs, state));
                batchCount++;
            }
            
            let avg_grad = DQN.divideGrad(grad1, batchCount);
            policy.updateParams(avg_grad, lr = 0.001);
            target.softUpdateFrom(policy);
            
            // env.render(ctx, canvas);
            // // console.log(new_state);
            // requestAnimationFrame(animate);

            total_reward += obs.reward;
            stepCount++;
        }


        if (env.done){

            console.log(total_reward);
            console.log(stepCount);
        }
    }
        

}
for (let i = 0; i < num_episodes; i++){
    const env = new CarRLEnvironment(-5000, "Keys");
    let obs = env.resetEnvironment();
    let state = obs.sensors;
    animate(env, obs, state);
}
