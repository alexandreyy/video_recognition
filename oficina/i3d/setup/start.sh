#!/bin/bash

mappings=()

echo Diretórios mapeados:

for param in ${@}
do
    mappings+="-v "${param}":"${param}" "
    echo ${param}
done

container_name="i3d"

rm_check="$(docker ps --all --quiet --filter=name="$container_name")"

if [ -n "$rm_check" ]; then
    docker stop $container_name > /dev/null 2>&1 && docker rm $container_name > /dev/null 2>&1
fi

xhost +local:root > /dev/null 2>&1

nvidia-docker run -it -v `pwd`/../src:/home/i3d/src -v /tmp/.X11-unix:/tmp/.X11-unix:ro ${mappings} -e DISPLAY=$DISPLAY --privileged --name $container_name $container_name /bin/bash