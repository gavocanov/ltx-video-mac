#!/bin/bash
xcodebuild -scheme LTXVideoGenerator \
    -destination 'platform=macOS' \
    -configuration Release \
    -derivedDataPath build \
    DEVELOPMENT_TEAM="" \
    CODE_SIGN_STYLE=Manual \
    CODE_SIGN_IDENTITY="-"
