FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    slurm-client \
    slurm-wlm-doc \
    slurmd \
    openssh-client \
    slurmctld \
    slurm-wlm-doc \
    mailutils \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    less syslog-ng-core \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    nano vim pwgen \
    && rm -rf /var/lib/apt/lists/*

# Module management system
RUN apt-get update && apt-get install -y \
    lua5.3 \
    lua-bit32:amd64 \
    lua-posix:amd64 \
    lua-posix-dev \
    liblua5.3-0:amd64 \
    liblua5.3-dev:amd64 \
    tcl \
    tcl-dev \
    tcl8.6 \
    tcl8.6-dev:amd64 \
    libtcl8.6:amd64 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://sourceforge.net/projects/lmod/files/Lmod-8.7.tar.bz2 && \
    tar -xf Lmod-8.7.tar.bz2 && cd Lmod-8.7 && \
    ./configure --prefix=/opt/apps && make install && \
    rm -r /Lmod-8.7* && \
    echo "/opt/apps/lmod/lmod/init/modulefiles/\*" > /opt/apps/lmod/lmod/init/.modulespath && \
    ln -s /opt/apps/lmod/lmod/init/profile /etc/profile.d/z00_lmod.sh

# Install and configure specific modules
RUN apt-get update && apt-get install -y openmpi-bin

RUN apt-get update && apt-get install -y \
    python3-pip && \
    pip install numpy && \
    pip install datetime && \
    pip install mpi4py

RUN mkdir /opt/apps/lmod/lmod/modulefiles/gcc
RUN mkdir /opt/apps/lmod/lmod/modulefiles/python
RUN mkdir /opt/apps/lmod/lmod/modulefiles/openmpi
RUN echo \
"\
whatis(\"Version: 9.4.0\")\n\
prepend_path( \"PATH\", \"/usr/bin/\")\n\
prepend_path( \"LD_LIBRARY_PATH\",\"/usr/lib/gcc/x86_64-linux-gnu/9\")\n\
set_shell_function(\"gcc\", \"/usr/bin/gcc-9\", \"/usr/bin/gcc-9\")\
" \
    > /opt/apps/lmod/lmod/modulefiles/gcc/9.4.0.lua
RUN echo \
"\
whatis(\"Version: 8.4.0\")\n\
prepend_path( \"PATH\", \"/usr/bin/\")\n\
prepend_path( \"LD_LIBRARY_PATH\",\"/usr/lib/gcc/x86_64-linux-gnu/8\")\n\
set_shell_function(\"gcc\", \"/usr/bin/gcc-8\", \"/usr/bin/gcc-8\")\
"\
    > /opt/apps/lmod/lmod/modulefiles/gcc/8.4.0.lua
RUN echo \
"\
whatis(\"Version: 3.8.10\")\n\
prepend_path( \"PATH\", \"/usr/bin/\")\n\
prepend_path( \"LD_LIBRARY_PATH\",\"/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/\")\n\
set_shell_function(\"python\", \"/usr/bin/python3 \$@\", \"/usr/bin/python3 \$*\")\
"\
    > /opt/apps/lmod/lmod/modulefiles/python/3.8.10.lua
RUN echo \
"\
whatis(\"Version: 4.0.3\")\n\
prepend_path( \"PATH\",           \"/usr/bin/\")\n\
prepend_path( \"LD_LIBRARY_PATH\",\"/usr/lib/x86_64-linux-gnu/\")\
"\
    > /opt/apps/lmod/lmod/modulefiles/openmpi/4.0.3.lua

COPY bin/docker-entrypoint.sh /etc/slurm-llnl/
COPY etc/slurm.conf /etc/slurm-llnl/
COPY examples /usr/share/slurm-examples

RUN useradd -ms /bin/bash user

RUN mkdir /state
RUN chown slurm:slurm /state

RUN mkdir /scratch
RUN chown slurm:slurm /state

RUN mkdir /scratch/user
RUN chown user:user /scratch/user

# Configure user env for Lmod
RUN echo "\nexport MODULEPATH=/opt/apps/lmod/lmod/modulefiles/" >> /home/user/.bashrc

# Simulation code and submission scripts
RUN echo \
"\
import numpy as np\n\
import sys\n\
import datetime\n\
\n\
def inside_circle(total_count):\n\
    x = np.random.uniform(size=total_count)\n\
    y = np.random.uniform(size=total_count)\n\
    radii = np.sqrt(x * x + y * y)\n\
    count = len(radii[np.where(radii<=1.0)])\n\
    return count\n\
\n\
def main():\n\
    n_samples = int(sys.argv[1])\n\
    start_time = datetime.datetime.now()\n\
    counts = inside_circle(n_samples)\n\
    end_time = datetime.datetime.now()\n\
    elapsed_time = (end_time - start_time).total_seconds()\n\
    my_pi = 4.0 * counts / n_samples\n\
    #print(my_pi)\n\
    size_of_float = np.dtype(np.float64).itemsize\n\
    memory_required = 3 * n_samples * size_of_float / (1024**3)\n\
    print(\"Pi: {}, memory: {} GiB, time: {} s\".format(my_pi, memory_required,\n\
                                                  elapsed_time))\n\
\n\
if __name__ == '__main__':\n\
    main()\n\
"\
    > /home/user/pi-serial.py

RUN echo \
"\
import numpy as np\n\
import sys\n\
import datetime\n\
from mpi4py import MPI\n\
\n\
def inside_circle(total_count):\n\
    host_name = MPI.Get_processor_name()\n\
    print(\"Rank {} generating {:n} samples on host {}.\".format(\n\
            rank, total_count, host_name))\n\
    x = np.float64(np.random.uniform(size=total_count))\n\
    y = np.float64(np.random.uniform(size=total_count))\n\
\n\
    radii = np.sqrt(x*x + y*y)\n\
\n\
    count = len(radii[np.where(radii<=1.0)])\n\
\n\
    return count\n\
\n\
if __name__ == '__main__':\n\
    comm = MPI.COMM_WORLD\n\
    cpus = comm.Get_size()\n\
    rank = comm.Get_rank()\n\
\n\
    if len(sys.argv) > 1:\n\
        n_samples = int(sys.argv[1])\n\
    else:\n\
        n_samples = 8738128 # trust me, this number is not random :-)\n\
\n\
    if rank == 0:\n\
        start_time = datetime.datetime.now()\n\
        print(\"Generating {:n} samples.\".format(n_samples))\n\
        partitions = [ int(n_samples / cpus) ] * cpus\n\
        counts = [ int(0) ] * cpus\n\
    else:\n\
        partitions = None\n\
        counts = None\n\
\n\
    partition_item = comm.scatter(partitions, root=0)\n\
    count_item = comm.scatter(counts, root=0)\n\
\n\
    count_item = inside_circle(partition_item)\n\
\n\
    counts = comm.gather(count_item, root=0)\n\
\n\
    if rank == 0:\n\
        my_pi = 4.0 * sum(counts) / n_samples\n\
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()\n\
\n\
        size_of_float = np.dtype(np.float64).itemsize\n\
        memory_required = 3 * n_samples * size_of_float / (1024**3)\n\
\n\
        pi_specific = np.pi\n\
        accuracy = 100*(1-my_pi/pi_specific)\n\
\n\
        summary = \"{:d},{:d},{:f},{:f},{:f}\"\n\
        print(\"Pi: {}, memory: {} GiB, time: {} s\".format(my_pi, memory_required,\n\
                                                  elapsed_time))\n\
"\
    > /home/user/pi-parallel.py

RUN echo \
"#!/usr/bin/env bash\
\n#SBATCH -J serial-pi\
\n#SBATCH -N 1\
\n#SBATCH -n 1\
\n#SBATCH -o %j.out\
\n#SBATCH -e %j.err\
\n\
\n# Load the computing environment we need\
\nmodule load python\
\n\
\n# Execute the task\
\npython3 pi-serial.py 4000000\
" \
    > /home/user/sub-serial.sh

RUN echo \
"#!/usr/bin/env bash\
\n#SBATCH -J parallel-pi\
\n#SBATCH -N 1\
\n#SBATCH -n 2\
\n#SBATCH -o %j.out\
\n#SBATCH -e %j.err\
\n\
\n# Load the computing environment we need\
\nmodule load python\
\nmodule load openmpi\
\n\
\n# Execute the task\
\nmpirun -np 2 python3 pi-parallel.py 4000000\
" \
    > /home/user/sub-parallel.sh

RUN chown user:user /home/user/*

CMD ["/etc/slurm-llnl/docker-entrypoint.sh"]
