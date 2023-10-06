# weighted-sims

## Para colocar no ar

Execute o comando: docker-compose up

## Caso mude o programa

Execute os comandos:

docker stop \$(docker ps -a -q)

docker rm \$(docker ps -a -q)

docker rmi -f \$(docker images -aq)
