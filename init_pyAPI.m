% init_pyAPI.m

pe = pyenv;
if pe.Status ~= 'NotLoaded'
  flag = 'Previous pyenv loaded; restart MATLAB';
  error(flag)
else
  % Set appropriate environment
  py_path = 'C:\Users\brian\Anaconda3\envs\tf_env\python.exe';
  pyversion(py_path);

  % Allow OutofProcess Python API
  pyenv('ExecutionMode', 'OutOfProcess');
end
