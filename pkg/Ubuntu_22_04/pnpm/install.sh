wget -qO- https://get.pnpm.io/install.sh | sh -

echo '# pnpm'                                               >> ~/.paths
echo 'export PNPM_HOME="/home/pga_gpc_3/.local/share/pnpm"' >> ~/.paths
echo 'case ":$PATH:" in'                                    >> ~/.paths
echo '  *":$PNPM_HOME:"*) ;;'                               >> ~/.paths
echo '  *) export PATH="$PNPM_HOME:$PATH" ;;'               >> ~/.paths
echo 'esac'                                                 >> ~/.paths
echo '# pnpm end'                                           >> ~/.paths
