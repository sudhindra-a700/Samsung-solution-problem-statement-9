# User Manual - SAC-GRU Traffic Analyzer

## Table of Contents
1. [Getting Started](#getting-started)
2. [Application Overview](#application-overview)
3. [User Interface Guide](#user-interface-guide)
4. [Features and Functionality](#features-and-functionality)
5. [Integration Guide](#integration-guide)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)
8. [Support](#support)

## Getting Started

### System Requirements
- **Android Version**: 7.0 (API level 24) or higher
- **RAM**: 4GB minimum, 6GB recommended
- **Storage**: 100MB available space
- **Processor**: ARM64 or x86_64 architecture
- **Network**: Wi-Fi or cellular data connection

### Installation

#### Method 1: APK Installation (Recommended)
1. Download the `SAC-GRU-Traffic-Analyzer.apk` file
2. Enable "Install from unknown sources" in Android Settings
3. Tap the APK file to install
4. Grant necessary permissions when prompted
5. Launch the app from your app drawer

#### Method 2: Development Installation
1. Connect your Android device to a computer
2. Enable USB debugging in Developer Options
3. Use Android Studio or ADB to install the app
4. Launch the app once installation completes

### First Launch Setup
1. **Welcome Screen**: Review the app introduction
2. **Permissions**: Grant necessary permissions:
   - Network access (for traffic analysis)
   - Storage access (for model and data storage)
3. **Model Loading**: Wait for the AI model to load (5-10 seconds)
4. **Ready to Use**: The app is now ready for traffic analysis

## Application Overview

### What is SAC-GRU Traffic Analyzer?

The SAC-GRU Traffic Analyzer is an AI-powered mobile application that uses advanced machine learning to classify network traffic in real-time. It combines **Soft Actor-Critic (SAC) reinforcement learning** with **Gated Recurrent Units (GRU)** to distinguish between different types of network traffic, particularly focusing on identifying REEL content (social media videos) vs NON-REEL content.

### Key Capabilities
- **Real-time Traffic Classification**: Analyze network traffic as it happens
- **High Accuracy**: 95%+ classification accuracy
- **Fast Processing**: Sub-5ms inference times
- **Social Media Integration**: Direct integration with YouTube and Instagram
- **Battery Efficient**: Minimal impact on device battery life
- **Privacy-First**: All processing happens on your device

## User Interface Guide

### Main Screen Layout

#### Navigation Structure
```
┌─────────────────────────────────────┐
│           Top App Bar               │
├─────────────────────────────────────┤
│                                     │
│         Content Area                │
│                                     │
├─────────────────────────────────────┤
│        Bottom Navigation            │
│  [Home] [Test] [Results] [Settings] │
└─────────────────────────────────────┘
```

#### Home Tab
The main dashboard showing:
- **Model Status Card**: Current AI model information
- **Quick Stats**: Recent classification statistics
- **Integration Cards**: YouTube and Instagram shortcuts
- **Performance Metrics**: Real-time performance indicators

#### Test Tab
Interactive testing interface:
- **Single Test Button**: Run individual classification tests
- **Batch Testing**: Run multiple tests simultaneously
- **Custom Input**: Test with custom network data
- **Test History**: View previous test results

#### Results Tab
Comprehensive results display:
- **Classification Results**: Detailed prediction outcomes
- **Confidence Scores**: AI confidence levels
- **Performance Metrics**: Speed and accuracy statistics
- **Export Options**: Save or share results

#### Settings Tab
Configuration options:
- **Model Settings**: AI model configuration
- **Performance Options**: Speed vs accuracy trade-offs
- **Integration Settings**: YouTube/Instagram preferences
- **About Information**: App version and details

### Visual Elements

#### Status Indicators
- 🟢 **Green**: System ready, model loaded
- 🟡 **Yellow**: Processing, loading in progress
- 🔴 **Red**: Error state, attention required
- ⚪ **Gray**: Inactive or disabled

#### Result Cards
```
┌─────────────────────────────────────┐
│  Classification Result              │
├─────────────────────────────────────┤
│  Prediction: REEL                   │
│  Confidence: 87.3%                  │
│  Inference Time: 2.4ms              │
│  Timestamp: 14:32:15                │
└─────────────────────────────────────┘
```

## Features and Functionality

### Core Features

#### 1. Real-Time Traffic Classification
**Purpose**: Analyze network traffic patterns to identify content types

**How to Use**:
1. Navigate to the **Test** tab
2. Tap **"Run Test"** button
3. The app will analyze current network traffic
4. Results appear in 2-5 seconds
5. View detailed classification results

**Understanding Results**:
- **REEL**: Social media video content (YouTube Shorts, Instagram Reels, TikTok)
- **NON-REEL**: Regular web traffic (browsing, downloads, streaming)
- **Confidence**: AI certainty level (0-100%)
- **Inference Time**: Processing speed in milliseconds

#### 2. Batch Testing
**Purpose**: Run multiple classifications for comprehensive analysis

**How to Use**:
1. Go to **Test** tab
2. Tap **"Batch Test"** button
3. Select number of tests (10, 50, 100)
4. Wait for completion
5. Review aggregate results

**Batch Results Include**:
- Average accuracy across all tests
- Performance distribution
- Error analysis
- Speed metrics

#### 3. Performance Monitoring
**Purpose**: Track app performance and system health

**Metrics Displayed**:
- **Inference Speed**: Average processing time
- **Memory Usage**: Current RAM consumption
- **Battery Impact**: Power usage statistics
- **Model Accuracy**: Classification success rate

#### 4. Social Media Integration
**Purpose**: Direct integration with popular social platforms

**YouTube Integration**:
1. Tap the **YouTube** card on home screen
2. App opens YouTube Shorts directly
3. Share YouTube URLs to the app for analysis
4. Automatic REEL detection for YouTube content

**Instagram Integration**:
1. Tap the **Instagram** card on home screen
2. App opens Instagram Reels section
3. Share Instagram Reel URLs for analysis
4. Automatic classification of Instagram content

### Advanced Features

#### 1. Custom Data Input
**Purpose**: Test the AI model with your own network data

**Steps**:
1. Navigate to **Test** → **Custom Input**
2. Enter network parameters manually:
   - Packet size
   - Protocol type
   - Port numbers
   - Timing information
3. Tap **"Analyze"**
4. Review custom classification results

#### 2. Export and Sharing
**Purpose**: Save and share analysis results

**Export Options**:
- **JSON Format**: Machine-readable data
- **CSV Format**: Spreadsheet-compatible
- **PDF Report**: Human-readable summary
- **Share Link**: Quick sharing with others

**How to Export**:
1. Go to **Results** tab
2. Select results to export
3. Tap **"Export"** button
4. Choose format and destination

#### 3. Historical Analysis
**Purpose**: Review past classifications and trends

**Features**:
- **Timeline View**: Chronological result display
- **Trend Analysis**: Pattern identification over time
- **Filtering Options**: Filter by date, type, or confidence
- **Statistics Dashboard**: Aggregate performance metrics

## Integration Guide

### YouTube Integration

#### Setup
1. Ensure YouTube app is installed on your device
2. Enable deep linking in Android settings
3. Grant necessary permissions for app switching

#### Usage Scenarios

**Scenario 1: Direct Launch**
1. Open SAC-GRU Traffic Analyzer
2. Tap **YouTube** integration card
3. App launches YouTube Shorts directly
4. Browse and interact with content normally

**Scenario 2: URL Analysis**
1. Copy a YouTube Shorts URL
2. Open SAC-GRU Traffic Analyzer
3. Paste URL in analysis field
4. Tap **"Analyze URL"**
5. View classification results

**Scenario 3: Share Integration**
1. In YouTube app, tap **Share** on any video
2. Select **SAC-GRU Traffic Analyzer** from share menu
3. App automatically analyzes the shared content
4. Results display immediately

### Instagram Integration

#### Setup
1. Install Instagram app on your device
2. Log in to your Instagram account
3. Enable app linking permissions

#### Usage Scenarios

**Scenario 1: Reels Browser**
1. Open SAC-GRU Traffic Analyzer
2. Tap **Instagram** integration card
3. App opens Instagram Reels section
4. Browse Reels with automatic analysis

**Scenario 2: Share Analysis**
1. In Instagram, find a Reel you want to analyze
2. Tap the **Share** button
3. Select **SAC-GRU Traffic Analyzer**
4. App analyzes the Reel content
5. View detailed classification results

### Deep Link Configuration

#### Supported URL Schemes
```
Custom Schemes:
- sacgru://analyze?url=[URL]
- sacgru://test?type=[TEST_TYPE]
- sacgru://results?id=[RESULT_ID]

Supported URLs:
- youtube.com/shorts/*
- youtu.be/*
- instagram.com/reel/*
- instagram.com/reels/*
```

#### Integration Testing
1. Test deep links using ADB:
   ```bash
   adb shell am start -W -a android.intent.action.VIEW -d "sacgru://analyze?url=https://youtube.com/shorts/example"
   ```
2. Verify proper app launching and URL handling
3. Test with various URL formats

## Troubleshooting

### Common Issues

#### Issue: App Won't Start
**Symptoms**: App crashes immediately after launch

**Solutions**:
1. **Restart Device**: Simple reboot often resolves issues
2. **Clear App Cache**: Settings → Apps → SAC-GRU → Storage → Clear Cache
3. **Reinstall App**: Uninstall and reinstall the application
4. **Check Android Version**: Ensure Android 7.0+ compatibility
5. **Free Storage Space**: Ensure 100MB+ available storage

#### Issue: Model Loading Failed
**Symptoms**: "Model not loaded" error message

**Solutions**:
1. **Check Internet Connection**: Initial model download requires internet
2. **Restart App**: Close and reopen the application
3. **Clear App Data**: Settings → Apps → SAC-GRU → Storage → Clear Data
4. **Verify Permissions**: Ensure storage permissions are granted
5. **Contact Support**: If issue persists, report to support team

#### Issue: Slow Performance
**Symptoms**: Long inference times (>10 seconds)

**Solutions**:
1. **Close Background Apps**: Free up system resources
2. **Restart Device**: Clear memory and refresh system
3. **Check Device Specs**: Verify minimum requirements
4. **Update App**: Ensure latest version is installed
5. **Reduce Concurrent Operations**: Avoid running multiple tests simultaneously

#### Issue: Integration Not Working
**Symptoms**: YouTube/Instagram integration fails

**Solutions**:
1. **Update Target Apps**: Ensure YouTube/Instagram are updated
2. **Check Permissions**: Verify app linking permissions
3. **Clear Default Apps**: Reset default app associations
4. **Test Deep Links**: Manually test URL schemes
5. **Reinstall Integration**: Clear and reconfigure integrations

### Error Codes

#### Error Code Reference
| Code | Description | Solution |
|------|-------------|----------|
| **E001** | Model loading failed | Clear cache and restart |
| **E002** | Insufficient memory | Close background apps |
| **E003** | Network timeout | Check internet connection |
| **E004** | Invalid input data | Verify input format |
| **E005** | Permission denied | Grant required permissions |
| **E006** | Storage full | Free up device storage |
| **E007** | Unsupported device | Check compatibility |
| **E008** | Integration failed | Reconfigure app linking |

### Performance Optimization

#### Optimizing Inference Speed
1. **Enable Hardware Acceleration**: 
   - Settings → Performance → Enable NNAPI
   - Settings → Performance → Enable GPU Acceleration
2. **Adjust Thread Count**:
   - Settings → Performance → Inference Threads (2-4 recommended)
3. **Close Unnecessary Apps**:
   - Free up CPU and memory resources
4. **Use Power Saving Mode Wisely**:
   - Disable aggressive power saving during analysis

#### Optimizing Battery Life
1. **Reduce Analysis Frequency**:
   - Use batch testing instead of continuous monitoring
2. **Enable Battery Optimization**:
   - Settings → Battery → Optimize for battery life
3. **Use Wi-Fi When Possible**:
   - Wi-Fi uses less power than cellular data
4. **Close App When Not in Use**:
   - Prevent background processing

## FAQ

### General Questions

**Q: What does SAC-GRU stand for?**
A: SAC-GRU stands for Soft Actor-Critic Gated Recurrent Unit, which describes the AI architecture combining reinforcement learning (SAC) with recurrent neural networks (GRU).

**Q: Is my data safe and private?**
A: Yes, all processing happens locally on your device. No personal data or network traffic is sent to external servers.

**Q: How accurate is the classification?**
A: The system achieves 95%+ accuracy on diverse network traffic patterns, with even higher accuracy on social media content.

**Q: Does the app work offline?**
A: Yes, once the AI model is loaded, the app works completely offline. Internet is only needed for initial setup and updates.

### Technical Questions

**Q: What types of network traffic can it analyze?**
A: The app can analyze various traffic types including HTTP/HTTPS web traffic, streaming media, social media content, and general internet communications.

**Q: How fast is the analysis?**
A: Analysis typically completes in 2-5 milliseconds, making it suitable for real-time applications.

**Q: Can I use this for network security?**
A: While designed for traffic classification, the technology can be adapted for security applications. However, this version focuses on content type identification.

**Q: What's the difference between REEL and NON-REEL?**
A: REEL refers to short-form video content (YouTube Shorts, Instagram Reels, TikTok), while NON-REEL includes regular web browsing, downloads, and other internet activities.

### Integration Questions

**Q: Why doesn't YouTube integration work?**
A: Ensure you have the latest YouTube app installed and that app linking permissions are enabled in Android settings.

**Q: Can I add other social media platforms?**
A: The current version supports YouTube and Instagram. Future updates may include additional platforms based on user demand.

**Q: How do I share results with others?**
A: Use the Export feature in the Results tab to generate shareable reports in various formats.

## Support

### Getting Help

#### In-App Support
- **Help Section**: Settings → Help & Support
- **Error Reporting**: Automatic crash reports with user consent
- **Feedback**: Settings → Send Feedback

#### Online Resources
- **Documentation**: Complete technical documentation available
- **Video Tutorials**: Step-by-step usage guides
- **Community Forum**: User discussions and tips
- **Developer Blog**: Updates and technical insights

#### Contact Information
- **Email Support**: support@sac-gru-analyzer.com
- **Bug Reports**: bugs@sac-gru-analyzer.com
- **Feature Requests**: features@sac-gru-analyzer.com
- **General Inquiries**: info@sac-gru-analyzer.com

#### Response Times
- **Critical Issues**: 24 hours
- **General Support**: 48-72 hours
- **Feature Requests**: 1-2 weeks
- **Bug Reports**: 24-48 hours

### Reporting Issues

#### Information to Include
When reporting issues, please provide:
1. **Device Information**: Model, Android version, RAM
2. **App Version**: Found in Settings → About
3. **Error Details**: Screenshots, error codes, steps to reproduce
4. **Network Environment**: Wi-Fi vs cellular, connection speed
5. **Usage Context**: What you were trying to do when the issue occurred

#### Bug Report Template
```
Device: [Samsung Galaxy S23]
Android Version: [13]
App Version: [1.0.0]
Error Code: [E001]

Steps to Reproduce:
1. Open app
2. Navigate to Test tab
3. Tap "Run Test"
4. Error appears

Expected Behavior:
Test should complete successfully

Actual Behavior:
App shows "Model loading failed" error

Additional Notes:
Issue started after recent Android update
```

### Updates and Maintenance

#### Update Schedule
- **Minor Updates**: Monthly (bug fixes, performance improvements)
- **Major Updates**: Quarterly (new features, UI improvements)
- **Security Updates**: As needed (immediate for critical issues)
- **Model Updates**: Bi-annually (AI model improvements)

#### Update Process
1. **Automatic Updates**: Enable in Google Play Store settings
2. **Manual Updates**: Check Google Play Store regularly
3. **Beta Testing**: Join beta program for early access
4. **Rollback**: Contact support if update causes issues

---

**Thank you for using SAC-GRU Traffic Analyzer!** This manual will help you get the most out of your AI-powered network analysis experience. For additional support or questions not covered here, please don't hesitate to contact our support team.

