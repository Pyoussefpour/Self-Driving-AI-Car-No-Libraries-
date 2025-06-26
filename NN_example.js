// create policy + target networks
const policy = new DQN([5,16,4]);
const target = policy.clone();   // target net

const buffer = new ReplayBuffer(10000);
const γ = 0.99;                  // discount

/* add a transition */
buffer.push(state, action, reward, nextState, done);

/* training step */
if (buffer.size() > 64) {
    const batch = buffer.sample(64);

    // Build training arrays
    const states   = batch.map(t => t.s);
    const targets  = batch.map(t => {
        const q    = policy.forward(t.s);
        const qNext= target.forward(t.s2);
        const y    = q.slice();                       // copy
        y[t.a] = t.r + (t.done ? 0 : γ * Math.max(...qNext));  // Bellman
        return y;
    });

    policy.fit(states, targets, 0.001);
    target.softUpdateFrom(policy, 0.005);   // τ = 0.005
}