# 2019-5-21 20:12:16
# Personal implementation of BM3D Denoising Algorithm

import numpy as np
import time
import cv2
from scipy.fftpack import dct, idct
from scipy.fftpack import fft, ifft
from matplotlib import pyplot as plt
__author__ = 'Prophet'

# Parameters

sigma = 25  # add-noise var
S1_blocksize = 8
S1_Threshold = 5000
S1_MaxMatch = 16
S1_DownScale = 1
S1_WindowSize = 50
lamb2d = 2.0
lamb3d = 2.7

S2_Threshold = 400
S2_MaxMatch = 32
S2_blocksize = 8
S2_DownScale = 1
S2_WindowSize = 75

beta = 2.0
A = 255

# Utils


def signalGen(A=255, f=10, fs=512, show=1):
    '''
    Complex Signal Demo
    '''
    phi = np.random.rand() * np.pi * 2
    t = np.linspace(0, 2 * np.pi * f, fs)
    sig = A * np.exp(1j * (t + phi))
    print(phi)
    if show:
        plt.subplot(2, 3, 1)
        plt.plot(np.real(sig))
        plt.title('real part')

        plt.subplot(2, 3, 2)
        plt.plot(np.imag(sig))
        plt.title('imag part')

        plt.subplot(2, 3, 3)
        plt.plot(np.abs(sig))
        plt.title('abs')
        # plt.show()
    return sig


def AWGN(sigma, signal, show=1):
    '''
    Add White Gauss Noise
    '''
    noise = sigma * np.random.randn(len(signal)) + \
        sigma * 1j * np.random.randn(len(signal))
    noise = noise / np.sqrt(2)
    if show:

        plt.subplot(2, 3, 4)
        plt.plot(np.real(signal + noise))
        plt.title('real part')

        plt.subplot(2, 3, 5)
        plt.plot(np.imag(signal + noise))
        plt.title('imag part')

        plt.subplot(2, 3, 6)
        plt.plot(np.abs(signal + noise))
        plt.title('abs')

        plt.show()
    return signal + noise, noise


def cvInit():
    '''
    actually dont use cv
    '''
    print('BM3D Initializing at {}'.format(time.time()))
    print('Initializing OpenCV')
    start = time.time()
    cv2.setUseOptimized(True)
    end = time.time()
    print('Initialized in {}s'.format(end - start))


def preFFT1D(noisySignal, blocksize, timer=True):
    s = time.time()
    blockFFT1D_all = np.zeros(
        (noisySignal.shape[0] - blocksize, blocksize), dtype=complex)
    for i in range(blockFFT1D_all.shape[0]):
        rBlock, iBlock = np.real(
            noisySignal[i:i + blocksize]), np.imag(noisySignal[i:i + blocksize])
        blockFFT1D_all[i, :] = fft(rBlock) + 1j * fft(iBlock)
        print(abs(fft(rBlock) + 1j * fft(iBlock)))
        exit()
    if timer:
        print('FFT all in {}s'.format(time.time() - s), end='\n')
    return blockFFT1D_all


def preDCT1D(noisySignal, blocksize, timer=True):
    s = time.time()
    blockDCT1D_all = np.zeros(
        (noisySignal.shape[0] - blocksize, blocksize), dtype=complex)
    for i in range(blockDCT1D_all.shape[0]):
        rBlock, iBlock = np.real(
            noisySignal[i:i + blocksize]), np.imag(noisySignal[i:i + blocksize])
        blockDCT1D_all[i, :] = dct(
            rBlock, norm='ortho') + 1j * dct(iBlock, norm='ortho')
    if timer:
        print('DCT all in {}s'.format(time.time() - s), end='\n')
    return blockDCT1D_all


def searchWindow1D(sig, RefPoint, blocksize, WindowSize):
    '''
        Set Boundary
    '''
    if blocksize >= WindowSize:
        print('Error: blocksize is smaller than WindowSize.\n')
        exit()
    Margin = np.zeros((2, 1), dtype=int)
    Margin[0, 0] = max(
        0, RefPoint + int((blocksize - WindowSize) / 2))  # left-top x
    Margin[1, 0] = Margin[0, 0] + WindowSize  # right-bottom x
    if Margin[1, 0] >= sig.shape[0]:
        Margin[1, 0] = sig.shape[0] - 1
        Margin[0, 0] = Margin[1, 0] - WindowSize

    return Margin


def computeSNR(signal, noise, estimate=None):
    if estimate is None:
        return 10 * np.log10(np.var(signal) / np.var(noise))
    else:
        return 10 * np.log10(np.var(signal) / np.var(estimate - signal))


def computeMSE(estimate, realsignal):
    if len(estimate) != len(realsignal):
        print('error calculating for different length')
        return 0
    else:
        return np.sum(np.abs(estimate - realsignal)**2) / len(realsignal)

# ===========================================================================


def S1_ComputeDist1D(BlockDCT1, BlockDCT2):
    """
    Compute the distance of two DCT arrays *BlockDCT1* and *BlockDCT2*
    """
    if BlockDCT1.shape != BlockDCT2.shape:
        print(
            'ERROR: two DCT Blocks are not at the same shape in step1 computing distance.\n')
        exit()
    blocksize = BlockDCT1.shape[0]
    if sigma > 40:
        ThreValue = lamb2d * sigma * 2
        BlockDCT1 = np.where(abs(BlockDCT1) < ThreValue, 0, BlockDCT1)
        BlockDCT2 = np.where(abs(BlockDCT2) < ThreValue, 0, BlockDCT2)
    return np.linalg.norm(BlockDCT1 - BlockDCT2)**2 / (blocksize**2)


def S1_Grouping1D(noisyImg, RefPoint, blockDCT1D_all, blocksize, ThreDist, MaxMatch, WindowSize):
    # initialization, get search boundary
    WindowLoc = searchWindow1D(noisyImg, RefPoint, blocksize, WindowSize)
    # print(WindowLoc)
    # number of searched blocks
    Block_Num_Searched = (WindowSize - blocksize + 1)
    # print(Block_Num_Searched)
    # 0 padding init
    BlockPos = np.zeros((Block_Num_Searched, 1), dtype=int)
    BlockGroup = np.zeros(
        (Block_Num_Searched, blocksize), dtype=complex)
    Dist = np.zeros(Block_Num_Searched, dtype=complex)

    RefDCT = blockDCT1D_all[RefPoint, :]
    # print(RefDCT)
    match_cnt = 0
    # k = []
    for i in range(WindowSize - blocksize + 1):
        SearchedDCT = blockDCT1D_all[WindowLoc[0, 0] + i, :]
        dist = S1_ComputeDist1D(RefDCT, SearchedDCT)
        if dist < ThreDist:
            BlockPos[match_cnt, :] = [
                WindowLoc[0, 0] + i, ]
            BlockGroup[match_cnt, :] = SearchedDCT
            Dist[match_cnt] = dist
            match_cnt += 1
    if match_cnt <= MaxMatch:
        BlockPos = BlockPos[:match_cnt, :]
        BlockGroup = BlockGroup[:match_cnt, :]
    else:
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)
        BlockPos = BlockPos[idx[:MaxMatch], :]
        BlockGroup = BlockGroup[idx[:MaxMatch], :]
    return BlockPos, BlockGroup


def S1_2DFiltering(BlockGroup):
    ThreValue = lamb3d * sigma
    nonzero_cnt = 0
    for i in range(BlockGroup.shape[1]):  # shape=(16,8)
        ThirdVector = dct(np.real(BlockGroup[:, i]), norm='ortho') + 1j * dct(
            np.imag(BlockGroup[:, i]), norm='ortho')  # 1D DCT
        ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.
        nonzero_cnt += np.nonzero(ThirdVector)[0].size
        BlockGroup[:, i] = list(idct(np.real(
            ThirdVector), norm='ortho') + 1j * idct(np.imag(ThirdVector), norm='ortho'))
    return BlockGroup, nonzero_cnt
    pass


def S1_Aggregation(BlockGroup, BlockPos, basic_estimate, basicWeight, basicKaiser, nonzero_cnt):

    if nonzero_cnt < 1:
        BlockWeight = 1.0 * basicKaiser
    else:
        BlockWeight = (1. / (sigma ** 2 * nonzero_cnt)) * basicKaiser
    # print(BlockPos.shape[0])
    for i in range(BlockPos.shape[0]):
        # print(BlockPos[i, 0], BlockPos[i, 0] + BlockGroup.shape[1])
        basic_estimate[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1]] += np.dot(BlockWeight, np.diag(
            idct(np.real(BlockGroup[i, :]), norm='ortho') + 1j * idct(np.imag(BlockGroup[i, :]), norm='ortho')))
        basicWeight[BlockPos[i, 0]: BlockPos[i, 0] +
                    BlockGroup.shape[1]] += BlockWeight


def BM3D1D_S1(noisySignal, para=None):
    # Using global variables
    basicEstimate = np.zeros(noisySignal.shape, dtype=complex)
    basicWeight = np.zeros(noisySignal.shape, dtype=complex)
    window = np.array(np.kaiser(S1_blocksize, beta) + 1j *
                      np.kaiser(S1_blocksize, beta), dtype=complex) / np.sqrt(2)
    # basicKaiser = np.array(window.T * window)
    basicKaiser = window

    blockDCT1D_all = preDCT1D(noisySignal, S1_blocksize)

    all_ = int((noisySignal.shape[0] - S1_blocksize) / S1_DownScale) + 2
    print('{} iterations remain.'.format(all_))
    count = 0
    start_ = time.time()

    for i in range(all_):
        if i != 0:
            print('i={}; Processing {}% ({}/{}), consuming {} s'.format(i,
                                                                        count * 100 / all_, count, all_, time.time() - start_))
        count += 1
        RefPoint = min(S1_DownScale * i,
                       noisySignal.shape[0] - S1_blocksize - 1)
        BlockPos, BlockGroup = S1_Grouping1D(
            noisySignal, RefPoint, blockDCT1D_all, S1_blocksize, S1_Threshold, S1_MaxMatch, S1_WindowSize)
        # print(BlockPos, BlockGroup)
        # exit()
        BlockGroup, nonzero_cnt = S1_2DFiltering(BlockGroup)
        S1_Aggregation(BlockGroup, BlockPos, basicEstimate,
                       basicWeight, basicKaiser, nonzero_cnt)
    basicWeight = np.where(basicWeight == 0, 1, basicWeight)
    basicEstimate[:] /= basicWeight[:]
    return basicEstimate


# ==========================================================

def S2_Aggregation1D(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, finalKaiser):
    """
    Compute the final estimate of the true-image by aggregating all of the obtained local estimates
    using a weighted average
    """
    BlockWeight = np.conj(WienerWeight) * finalKaiser
    for i in range(BlockPos.shape[0]):
        finalImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1]
                 ] += BlockWeight * (idct(np.real(BlockGroup_noisy[i, :]), norm='ortho') + 1j * idct(np.imag(BlockGroup_noisy[i, :]), norm='ortho'))
        finalWeight[BlockPos[i, 0]:BlockPos[i, 0] +
                    BlockGroup_noisy.shape[1]] += BlockWeight


def S2_2DFiltering(BlockGroup_basic, BlockGroup_noisy):
    """
    Wiener Filtering
    """
    Weight = 0
    coef = 1.0 / BlockGroup_noisy.shape[0]
    coef = coef
    for i in range(BlockGroup_noisy.shape[1]):
        Vec_basic = dct(np.real(BlockGroup_basic[:, i]), norm='ortho') + 1j * dct(
            np.imag(BlockGroup_basic[:, i]), norm='ortho')
        Vec_noisy = dct(np.real(BlockGroup_noisy[:, i]), norm='ortho') + 1j * dct(
            np.imag(BlockGroup_noisy[:, i]), norm='ortho')
        Vec_value = np.abs(Vec_basic) ** 2 * coef
        Vec_value /= (Vec_value + sigma ** 2)  # pixel weight
        Vec_noisy *= Vec_value
        Weight += np.sum(Vec_value)
        BlockGroup_noisy[:, i] = list(idct(
            np.real(Vec_noisy), norm='ortho') + 1j * idct(np.imag(Vec_noisy), norm='ortho'))
    if Weight > 0:
        WienerWeight = 1. / (sigma ** 2 * Weight)
    else:
        WienerWeight = (1.0 + 1j) / 1.414
    return BlockGroup_noisy, WienerWeight


def S2_ComputeDist1D(sig, Point1, Point2, BlockSize):
    """
    Compute distance between blocks whose left-top margins' coordinates are *Point1* and *Point2*
    """
    Block1 = (sig[Point1[0]:Point1[0] + BlockSize]).astype(complex)
    Block2 = (sig[Point2[0]:Point2[0] + BlockSize]).astype(complex)
    return np.linalg.norm(Block1 - Block2)**2 / (BlockSize**2)


def S2_Grouping1D(basicEstimate, noisyImg, RefPoint, BlockSize, ThreDist, MaxMatch, WindowSize,
                  BlockDCT_basic, BlockDCT_noisy):
    WindowLoc = searchWindow1D(basicEstimate, RefPoint, BlockSize, WindowSize)

    Block_Num_Searched = (WindowSize - BlockSize + 1)

    BlockPos = np.zeros((Block_Num_Searched, 1), dtype=int)
    BlockGroup_basic = np.zeros(
        (Block_Num_Searched, BlockSize), dtype=complex)
    BlockGroup_noisy = np.zeros(
        (Block_Num_Searched, BlockSize), dtype=complex)
    Dist = np.zeros(Block_Num_Searched, dtype=complex)
    match_cnt = 0
    for i in range(WindowSize - BlockSize + 1):
        SearchedPoint = [WindowLoc[0, 0] + i]
        dist = S2_ComputeDist1D(
            basicEstimate, [RefPoint], SearchedPoint, BlockSize)
        if dist < ThreDist:
            BlockPos[match_cnt, :] = SearchedPoint
            Dist[match_cnt] = dist
            match_cnt += 1
    if match_cnt <= MaxMatch:
        BlockPos = BlockPos[:match_cnt, :]
    else:
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)
        BlockPos = BlockPos[idx[:MaxMatch], :]
    for i in range(BlockPos.shape[0]):
        SimilarPoint = BlockPos[i, :]
        BlockGroup_basic[i, :] = BlockDCT_basic[SimilarPoint[0], :]
        BlockGroup_noisy[i, :] = BlockDCT_noisy[SimilarPoint[0], :]
    BlockGroup_basic = BlockGroup_basic[:BlockPos.shape[0], :]
    BlockGroup_noisy = BlockGroup_noisy[:BlockPos.shape[0], :]
    return BlockPos, BlockGroup_basic, BlockGroup_noisy


def BM3D1D_S2(noisySignal, basicEstimate):
    finalEstimate = np.zeros(noisySignal.shape, dtype=complex)
    finalWeight = np.zeros(noisySignal.shape, dtype=complex)
    Window = np.array(np.kaiser(S2_blocksize, beta) + 1j *
                      np.kaiser(S2_blocksize, beta), dtype=complex) / np.sqrt(2)
    finalKaiser = Window

    BlockDCT_noisy = preDCT1D(noisySignal, S2_blocksize)
    BlockDCT_basic = preDCT1D(basicEstimate, S2_blocksize)
    count = 0
    start_ = time.time()
    all_ = int((basicEstimate.shape[0] - S2_blocksize) / S2_DownScale) + 2

    for i in range(int((basicEstimate.shape[0] - S2_blocksize) / S2_DownScale) + 2):
        print('i={}; Processing {}% ({}/{}), consuming {} s'.format(i,
                                                                    count * 100 / all_, count, all_, time.time() - start_))
        RefPoint = min(S2_DownScale * i,
                       basicEstimate.shape[0] - S2_blocksize - 1)
        BlockPos, BlockGroup_basic, BlockGroup_noisy = S2_Grouping1D(basicEstimate, noisySignal,
                                                                     RefPoint, S2_blocksize,
                                                                     S2_Threshold, S2_MaxMatch,
                                                                     S2_WindowSize,
                                                                     BlockDCT_basic,
                                                                     BlockDCT_noisy)

        BlockGroup_noisy, WienerWeight = S2_2DFiltering(
            BlockGroup_basic, BlockGroup_noisy)
        S2_Aggregation1D(BlockGroup_noisy, WienerWeight, BlockPos, finalEstimate, finalWeight,
                         finalKaiser)
        count += 1

    finalWeight = np.where(finalWeight == 0, 1, finalWeight)
    finalEstimate[:] /= finalWeight[:]
    return finalEstimate
# ==========================================================


if __name__ == '__main__':
    cvInit()

    plt.figure()
    sig = signalGen(A)
    sig2 = signalGen(A / 2, f=5)
    sig3 = signalGen(A / 3, f=128)
    sig = sig + sig2 + sig3
    noisySignal, noise = AWGN(25, sig)
    print(np.var(noisySignal))

    S1_start = time.time()
    signalBasicEstimate = BM3D1D_S1(noisySignal)
    print('\nFinish Basic Estimate in {} s'.format(time.time() - S1_start))

    print('signal power:\t{}'.format(np.var(sig)))
    print('noise power:\t{}'.format(np.var(noise)))
    print('SNR:\t\t\t{} dB'.format(computeSNR(sig, noise)))
    print('basic estimate SNR:\t{} dB'.format(
        computeSNR(sig, None, signalBasicEstimate)))
    print('basic estimate MSE:{}'.format(
        computeMSE(signalBasicEstimate[:-1], sig[:-1])))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.real(signalBasicEstimate))
    plt.subplot(1, 3, 2)
    plt.plot(np.imag(signalBasicEstimate))
    plt.subplot(1, 3, 3)
    plt.plot(np.abs(signalBasicEstimate))
    plt.show()

    S2_start = time.time()
    finalEstimate = BM3D1D_S2(signalBasicEstimate, noisySignal)
    print('\nFinish Final Estimate in {} s'.format(time.time() - S2_start))
    print('signal power:\t{}'.format(np.var(sig)))
    print('noise power:\t{}'.format(np.var(noise)))
    print('SNR:\t\t\t{} dB'.format(computeSNR(sig, noise)))
    print('final estimate SNR:\t{} dB'.format(
        computeSNR(sig, None, finalEstimate)))
    print('final estimate MSE:{}'.format(
        computeMSE(finalEstimate[:-1], sig[:-1])))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.real(finalEstimate))
    plt.subplot(1, 3, 2)
    plt.plot(np.imag(finalEstimate))
    plt.subplot(1, 3, 3)
    plt.plot(np.abs(finalEstimate))
    plt.show()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.stem(np.abs(fft(np.real(sig)) + 1j * fft(np.imag(sig))))
    plt.subplot(1, 2, 2)
    plt.stem(np.abs(fft(np.real(finalEstimate)) +
                    1j * fft(np.imag(finalEstimate))))
    plt.show()
    exit()
    # Basic_PSNR = computePSNR()
