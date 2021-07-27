# parsec console
pkg install console
# Firewall rule
pkg install firewallrule
    echo moonligt > /tmp/name
    echo 47984, 47989, 48010 > /tmp/tcp
    echo 5353, 47998, 47999, 48000, 48002, 48010 > /tmp/udp
# server manager
pkg install lanserver
    server manager
    Add roles and features
    Features
    Wireless LAN Server
# acceleration3s script
pkg install acceleration3s
    download https://github.com/acceleration3/cloudgamestream/archive/refs/heads/master.zip
    modify step 1; remove if($InstallVideo) { to the end
    install with powershell as administrator
    Set-ExecutionPolicy Unrestricted; ./Setup.ps1
        change in ExecutionPolicy: all
        scripts that you trust: run once
        install VCABLE: No
        GRID Drivers: No
# activate and configure geforce experience
pkg install geforce_experience
    start geforce experience
    get started
    login
    settings
    Shield
    enable game stream
    Add
    C:\Windows\System32\mstsc.exe
# moonlight setup
pkg install moonlight
    download https://github.com/moonlight-stream/Internet-Streaming-Helper/releases/download/v5.5.3/InternetHostingToolSetup-v5.5.3.exe
    Agree and install
    Moonlight internet streaming tester

