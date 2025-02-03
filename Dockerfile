FROM debian:latest
VOLUME /var/log /downloads
EXPOSE 22/tcp
ARG all_proxy new_user user_id gid_pulse gid_pulse_access
COPY mppose.tgz ohmyzsh_install.sh /root/
RUN apt update && apt -y install sudo curl vim zsh tmux lrzsz net-tools iproute2 dnsutils git openssh-server locales python3 python3-pip python3-venv ffmpeg pulseaudio && groupmod -g $gid_pulse pulse && groupmod -g $gid_pulse_access pulse-access && useradd -s /usr/bin/zsh -d /home/$new_user -u $user_id $new_user && usermod -aG adm,sudo,video,pulse-access $new_user && echo "$new_user:helloworld" | chpasswd && echo "$new_user ALL=NOPASSWD: ALL" >> /etc/sudoers && mkdir -p /home/$new_user && mv /root/ohmyzsh_install.sh /home/$new_user/ && chown -R "$new_user:$new_user" /home/$new_user && sudo -u $new_user sh /home/$new_user/ohmyzsh_install.sh && sed -i 's/#PermitRootLogin .*$/PermitRootLogin yes/g' /etc/ssh/sshd_config && sed -i 's/^ZSH_THEME=.*$/ZSH_THEME="ys"/g' /home/$new_user/.zshrc && sed -i 's/^# \(zh_CN\|en_US\).UTF-8 UTF-8/\1.UTF-8 UTF-8/g' /etc/locale.gen && locale-gen && echo '\nexport LC_ALL="en_US.UTF-8"\nexport LANG="en_US.UTF-8"\nexport TZ="Asia/Shanghai"' >> /home/$new_user/.zshrc && echo "set ts=4\nset sw=4\nset nospell\nsyntax on\nset wrap\nset ai\nset fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936\nset termencoding=utf-8\nset encoding=utf-8\nset viminfo='1000,<500\n" > /home/$new_user/.vimrc && mkdir -p /home/$new_user/venv/src && python3 -m venv /home/$new_user/venv && . /home/$new_user/venv/bin/activate && cd /home/$new_user/venv/src && mv /root/mppose.tgz . && tar zxvf ./mppose.tgz && cd /home/$new_user/venv/src/mppose && pip install -r requirements.txt && chown -R "$new_user:$new_user" /home/$new_user && echo "\ndefault-server = unix:/run/user/1000/pulse/native" >> /etc/pulse/client.conf
ENTRYPOINT ["/usr/bin/zsh"]
