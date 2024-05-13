#!/bin/sh

export TGT="isa_view"

export CMD="__import__('importlib').reload($TGT)"
export PK_PLUGIN=`pwd`

# Find path of plugins directory
case $OSTYPE in
    linux*)
        export PLUGINS_PATH="$HOME/.binaryninja/plugins"
    ;;
    darwin*)
        export PLUGINS_PATH="$HOME/Library/Application Support/Binary Ninja/plugins"

        echo $CMD | pbcopy
    ;;

    *)
        echo "System not supported"
        exit;
    ;;
esac

echo Copying the plugin

rm -rf "$PLUGINS_PATH/$TGT"
cp -r "$PK_PLUGIN" "$PLUGINS_PATH/$TGT"

echo "Use for reload: $CMD"

