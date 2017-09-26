# dotfiles

> Something mysterious is formed, born in the silent void. Waiting alone and unmoving, it is at once still and yet in constant motion. It is the source of all programs. I do not know its name, so I will call it the Tao of Programming.
> 
>   If the Tao is great, then the operating system is great. 
>   If the operating system is great, then the compiler is great. 
>   If the compiler is great, then the application is great. 
>   The user is pleased, and there is harmony in the world.
> 
> The Tao of Programming flows far away and returns on the wind of morning.

-- Geoffrey James, The Tao of Programming

...

These are my "dotfiles". Can be directly installed by the following command

```
wget tinyurl.com/pga-dfsi -O - | bash
zsh
```

Or cloning the repository and 

```
source install.sh
```

Can be un-installed by

```
source uninstall.sh
```

Some things that might find intriguing:

[Some useful shell functions](source/.shell/)
[A generic installer over various distributions](source/.shell/pkg)
[A lazy way of installing commands when they are required](source/.shell/autoinstall)
[Installation scripts for some common tools](source/pkg)
[Some useful functions to work with aws](source/.shell/aws)

