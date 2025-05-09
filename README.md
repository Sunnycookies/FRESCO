## FRESCO相关文档（T10）

### 关于 `FRESCO` 、 `fresco` 和 `fresco_new` 三个文件夹的区别

- `FRESCO`：不是扩展实验的文件夹（应该是直接从Github上下载的）
- `fresco`：扩展实验前中期的文件夹，文件较多，重名文件较多
- `fresco_new`：扩展实验后期的文件夹，将 `fresco` 无用文件删除，将代码进行了整理

### 关于过去版本的备份

- 以 git 仓库的形式将一些关键实验节点备份在了网盘 `linjunxin/fresco/backup.git` 中。实际上 `fresco_new` 与仓库同步，可以通过 git 进行回溯查看

### 关于不同实验条件的设置与关闭

- 实际上，所有实验设置均在 ***config*** 提供，并经由 check_config() 加工、检查，最后传入视频处理函数
- 关于 ***config*** 的完整内容，详见 ***`config/ref_config.yaml`***
- 如果需要修改实验条件，可以直接通过修改 ***config*** 的方式进行；<u>因为 synth_mode 与 edit_mode 会涉及是否使用 ddim inversion，从而影响模型结构的问题，因此需要在加载模型前设置好；其余 ***config*** 可以在模型加载完后修改</u>
- 不考虑运行结果的存储，以下列出一些条件的 ***<u>最晚修改位置</u>***，若在该位置之后修改，则可能会造成运行过程中前后设置不一致

| 设置项          | 开关出现形式                         | 最晚修改位置                                          | 备注                                                         |
| --------------- | ------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| ebsynth         | ***run_ebsynth***                    | 加载模型前                                            |                                                              |
| tokenflow       | ***run_tokenflow***                  | 加载模型前                                            | 此外需要修改 ***config['use_inversion']=True***              |
| Mixed           | ***config['primary_select']***       | 加载模型前                                            | 该开关直接作用是开启锚点帧的选取。完整开启 Mixed 还需要同时设置 ebsynth 和 tokenflow（因此也需要设置 ***config['use_inversion']=True***） |
| SDEdit          | ***config['edit_mode']***            | 加载模型前                                            |                                                              |
| pnp             | ***config['edit_mode']***            | 加载模型前                                            | 此外需要修改 ***config['use_inversion']=True***              |
| fresco          | ***config['use_fresco']***           | ***`pipe_FRESCO.py`*** 中line 406，为fresco准备参数前 |                                                              |
| inference steps | ***config['num_inference_steps']***  | 加载模型前                                            |                                                              |
| keyframe mode   | ***config['keyframe_select_mode']*** | ***`run_fresco.py`*** 中line 157，即准备关键帧索引前  | 注意 check_config() 中的条件，即仅使用 ebsynth 时 keyframe 应当为 fixed |

### 关于FRESCO子模块的开关

- 原 FRESCO 的子模块开关在 inference step loop 之外；为了适配 tokenflow，现在FRESCO 的子模块开关在 inference step loop 内，总计有两处（关键帧编辑时、tokenflow 合成时）
- 为方便书写，以下 ***frescoProc.controller*** 简写为 ***ctrlr***
- 为方便书写，以下函数均不写出参数，实际使用时需注意参数设置
- 同时运行多个 ***`run_fresco.py`*** 时，请注意 ***config['temp_paras_save_path']*** 的设置，防止出现同一个目录的多次使用

| 子模块                    | 开启方式                  | 开关位置-关键帧编辑                                | 开关位置-tokenflow合成                             | 备注                                                         |      |
| ------------------------- | ------------------------- | -------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ | ---- |
| feature optimization      | apply_FRESCO_opt()        | ***`pipe_FRESCO.py`***  中line 451处               | ***`pipe_FRESCO.py`***  中line 578处               | 如需关闭，注释掉 apply_FRESCO_opt() 或使用disable_FRESCO_opt() 均可 |      |
| spatial guided attention  | ctrlr.enable_intraattn()  | ***`pipe_FRESCO.py`***  中line 456处               | ***`pipe_FRESCO.py`***  中line 583处               | 如需关闭，使用 ctrlr.disable_intraattn() 即可。<u>为了适配 loop 模式关键帧，intraattn 的存储机制拓展到了多维， ***ctrlr*** 中存有多组 intraattn 数据；如需保留存放的数据以便后续继续使用，则 ctrlr.disable_intraattn() 以及 ctrlr.enable_intraattn() 的参数中设置 ***reset=False***</u>。如需开启，须指明当前数据的组号，即设置 ***group_ind*** |      |
| cross frame attetion      | ctrlr.enable_cfattn()     | ***`pipe_FRESCO.py`***  中未使用，推荐在line 459处 | ***`pipe_FRESCO.py`***  中未使用，推荐在line 586处 | 如需关闭，使用ctrlr.disable_cfattn() 即可                    |      |
| temporal guided attention | ctrlr.enable_interattn()  | ***`pipe_FRESCO.py`***  中line 458处               | ***`pipe_FRESCO.py`***  中line 585处               | 如需关闭，使用 ctrlr.disable_interattn() 即可                |      |
| attention 统一开关        | ctrlr.enable_controller() | ***`pipe_FRESCO.py`***  中line 450处               | ***`pipe_FRESCO.py`***  中line 577处               | 如需关闭，使用 ctrlr.disable_controller() 即可。<u>注意intraattn相关参数的设置</u> |      |

### 关于新增的辅助脚本

- ***`test.py`***：相同实验条件下的自动化多视频 config 生成与 ***`run_fresco.py`*** 运行
- ***`test.sh`***：多种实验条件组合下的自动化多视频 config 生成与 ***`run_fresco.py`*** 运行
- ***`to_video_multi.py`***：用多个文件夹下的视频帧生成拼接视频。给定的文件夹须包含 `keys` 子文件夹，keys子文件夹内为待合成视频帧
- ***`auto_eb.py`***：用给定文件夹的 `keys` 子文件夹中的关键帧，以及给定视频，运行ebsynth。关键帧的序号须与视频中的帧序号一致
- ***`run_fresco.ipynb`***：***`run_fresco.py`*** 的Jupyter参考实现版本