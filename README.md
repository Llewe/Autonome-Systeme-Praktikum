# Autonome-Systeme-Praktikum

## Setup
1. Create env `conda create -n l-ki python=3.9`
2. Switch to env `conda activate l-ki` (depending on os)
3. Install packages `pip install -r requirements.txt`

### Setup Unity Env (Environment Executable)
> https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Executable.md

## Application Parameter
### Required:
- `-env <type>` set the enviroment (`gym` or `unity`)
- `-tag` give the run an individual tag/name 

### Optional
- `-d` use the demo ppo
- `-r` enable replay mode (turn graphics on)
- `-eval` run evaluation of trained model
- `-cpu` forces to use the cpu 
- `-env_n` name the domain name 
- `-agent` (ppo, random agent) set the agent type 

### Hyperparamters (if not given, default values are selected)
- `-e <number>` number of episodes in this run 
- `-us <number>` number of steps until update (n_steps/update_timestep)
- `-g <number>` discount factor
- `-lr_a <number>` learn rate of the actor
- `-lr_c <number>` learn rate of the critic
- `-ke <number>` "--k_epochs"
- `-e_clip <number>` eps_clip
- `-a_std <number>` action_std
- `-a_std_rate` action standard deviation decay rate 
- `-a_std_freq` action standard deviation decay frequency
- `-a_std_min` minimum action standard deviation

### Environment Parameter
Recommended parameter bounds: https://unity-technologies.github.io/ml-agents/Learning-Environment-Examples/#3dball-3d-balance-ball

- `--env_rngMass` default: `False` enable random mass
- `--env_mass <float>` default: `1.` set mass if random is disabled
- `--env_minMass <float>` default: `0.1` min mass if random is enabled
- `--env_maxMass <float>` default: `20.` max mass if random is enabled
- `--env_rngGravity` default: `False` enable random gravity
- `--env_gravity <float>` default: `9.81` set gravity if random is disabled
- `--env_minGravity <float>` default: `4.` min gravity if random is enabled
- `--env_maxGravity <float>` default: `105.` max gravity if random is enabled
- `--env_rngScale` default: `False` enable random ball scale
- `--env_scale <float>` default: `1.` set scale if random and individual scale is disabled
- `--env_minScale <float>` default: `0.2` min scale if random is enabled (works also combined with individual scale)
- `--env_maxScale <float>` default: `5.` max scale if random is enabled (works also combined with individual scale)
- `--env_video` default: `False` enable graphical output
- `--env_timeScale <float>` default: `20.` time speedup

> The Following parameters can be used in every enviroment but will only affect the behavior of a [modified](#Modifications-of-the-3DBall-domain) one.

- `--env_individualScale` default: `False` enable individual ball scale for x y z axis
- `--env_scale_x <float>` default: `1.` set scale for x axis if random scale is disabled and individual scale is enabled
- `--env_scale_y <float>` default: `1.` set scale for y axis if random scale is disabled and individual scale is enabled
- `--env_scale_z <float>` default: `1.` set scale for z axis if random scale is disabled and individual scale is enabled
- `--env_scale_max_dev <float>` default: `0.` set maximumum deviation of individual scales (if `0` no limit is set)
- `--env_color_r <float>` default: `0.85` set ball color (Red) value must be between [0-1]
- `--env_color_g <float>` default: `0.33` set ball color (Green) value must be between [0-1]
- `--env_color_b <float>` default: `0.93` set ball color (Blue) value must be between [0-1]
- `--env_color_a <float>` default: `1.` set ball color (Alpha) value must be between [0-1]
- `--env_bounciness <float>` default: `1.` set bounciness of the ball [0-1]
- `--env_dynamicFriction <float>` default: `0.` set dynamic friction of the ball [0-1] (more information: https://docs.unity3d.com/ScriptReference/PhysicMaterial.html)
- `--env_staticFriction <float>` default: `0.` set static friction of the ball [0-1] (more information: https://docs.unity3d.com/ScriptReference/PhysicMaterial.html)

## Check Carbon Footprint
Load webpage `carbonboard --filepath="./out/emissions.csv"`

## Check Model Logs - TensorBoard
```
tensorboard --logdir ./generated --port 9238
```

## Slurm
To create jobs the `jobList.sh` file must be modified. After every job is defined there the jobs can be send to the slum engine with the `training.sh` script. To start the script navigate into the slurm folder and run the following command.
```
sh training.sh
```
For clean up the log files of the slum engine will be deleted before starting a new batch of jobs with the `training.sh` script.

## Modifications of the 3DBall domain
1. Copy one of the Scenes from `ML-Agents/Examples/3DBall/Scenes/`, rename and open it
2. Copy a physic material from `ML-Agents/Examples/Soccer/Materials/Physic_Materials` to `ML-Agents/Examples/3DBall/Materials/Physic_Materials` and rename it to `My3DBall`.
3. Do the following steps for every ball in your scene:
   1. Removed `Sphere Collider` from the ball
   2. Added `Mesh Collider` to the ball (enable `Convex`, select the material `My3DBall` from `2.`, select Sphere as Mesh).
4. Copy and rename the `Ball3DAgent.cs` Script from `ML-Agents/Examples/3DBall/Scenes/`.
5. Edit the script as desired. For this project replace the `SetBall` function with
```C#
public void SetBall()
    {

    //Set the attributes of the ball by fetching the information from the academy
    ball.GetComponent<MeshCollider>().material.bounciness = m_ResetParams.GetWithDefault("bounciness", 1.0f);
    ball.GetComponent<MeshCollider>().material.dynamicFriction = m_ResetParams.GetWithDefault("dynamicFriction", 0.0f);
    ball.GetComponent<MeshCollider>().material.staticFriction = m_ResetParams.GetWithDefault("staticFriction", 0.0f);

    var color_r = m_ResetParams.GetWithDefault("color_r", 0.85f);
    var color_g = m_ResetParams.GetWithDefault("color_g", 0.33f);
    var color_b = m_ResetParams.GetWithDefault("color_b", 0.93f);
    var color_a = m_ResetParams.GetWithDefault("color_a", 1f);

    ball.GetComponent<Renderer>().material.color = new Color(color_r,color_g,color_b,color_a);

    m_BallRb.mass = m_ResetParams.GetWithDefault("mass", 1.0f);
    var scale_x = m_ResetParams.GetWithDefault("scale_x", 1.0f);
    var scale_y = m_ResetParams.GetWithDefault("scale_y", 1.0f);
    var scale_z = m_ResetParams.GetWithDefault("scale_z", 1.0f);
    var scale_max_deviation = m_ResetParams.GetWithDefault("scale_max_deviation", 0.0f);

    if (scale_max_deviation > 0f){
        var diff_xy = scale_x-scale_y;
        if(Math.Abs(diff_xy) > scale_max_deviation){
            scale_y =scale_x - Math.Sign(diff_xy) * scale_max_deviation;
        }
        var diff_xz = scale_x-scale_z;  
        if(Math.Abs(diff_xz) > scale_max_deviation){
            scale_z =scale_x - Math.Sign(diff_xz) * scale_max_deviation;
        }
        var diff_yz = scale_y-scale_z;  
        if(Math.Abs(diff_yz) > scale_max_deviation){
            scale_z =scale_y - Math.Sign(diff_yz) * scale_max_deviation;
        }
    }

    ball.transform.localScale = new Vector3(scale_x, scale_y, scale_z);

}
```
For this project also the following imports are needed.
```C#
using Random = UnityEngine.Random;
using System;
```
6. Save the Script
7. Do the following steps for every agent in your scene:
    1.  Remove `Ball 3D Agent (Script)`
    2.  Add the earlier modified script
8. Save everything
9. Don't forget to select the new sceen under `Build settings` of your project befor building it

