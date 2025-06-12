const canvas=document.getElementById("carCanvas");
canvas.width=200;

const ctx = canvas.getContext("2d");
const road=new Road(canvas.width/2,canvas.width*0.9);


const env = new CarRLEnvironment(-5000, "Keys");
let state = env.resetEnvironment();
let stepCount = 0;
let total_reward = 0;

function animate(){
    if (!env.done && stepCount < 10000){
        
        new_state = env.step(env.car.controls);
        env.render(ctx, canvas);
        // console.log(new_state);
        requestAnimationFrame(animate);

        total_reward += new_state.reward;
        stepCount++;
    }


    if (env.done){
        console.log(total_reward);
        console.log(stepCount);
    }


}

animate();

