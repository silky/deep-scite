{-# LANGUAGE OverloadedLists        #-}
{-# LANGUAGE ConstraintKinds        #-}
{-# LANGUAGE DataKinds              #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE OverloadedStrings      #-}
{-# LANGUAGE RankNTypes             #-}
{-# LANGUAGE ScopedTypeVariables    #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts      #-}
--
-- TODO:
--  Should all numbers be instances be able to be TF.scalars? Yes.
--
module Main where

import qualified TensorFlow.Ops             as TF
import qualified TensorFlow.GenOps.Core     as TFC
import qualified TensorFlow.Types           as TF
import qualified TensorFlow.Build           as TF
import qualified TensorFlow.BuildOp         as TF
import qualified TensorFlow.Session         as TF
import qualified TensorFlow.Tensor          as TF
import TensorFlow.Ops () -- Num instance for Tensor
import qualified TensorFlow.ControlFlow     as TF
import qualified TensorFlow.Gradient        as TF
import qualified TensorFlow.Nodes           as TF
import qualified TensorFlow.EmbeddingOps    as TF
import qualified TensorFlow.NN              as TF
import           Data.Csv                   ( FromNamedRecord(..)
                                            , decodeByName
                                            , (.:)
                                            )
import qualified Data.Vector                as V
import           Data.Int                   (Int32, Int64)
import           Data.Word                  (Word16)
import           Data.Monoid                (mconcat)
import           Data.List                  (genericLength)
import           Control.Monad              (join)
import           Control.Monad.IO.Class     (liftIO)
import           Lens.Family2               ((.~), (&))
import           Control.Monad              ( zipWithM
                                            , when
                                            , forM
                                            , forM_
                                            )
import qualified Data.ByteString.Lazy       as BL
import           Data.ByteString            (ByteString)
import           Control.Applicative        ( (<*>)
                                            , (<$>)
                                            )
import           Data.Text                  ( Text(..)
                                            , splitOn
                                            , unpack
                                            )
import           System.FilePath            ( (</>)
                                            , (<.>))

import Data.RVar (runRVar)
import Data.Random.Extras (shuffle)
import Data.Random.RVar ( RVar(..)
                        )
import Data.Random.Extras
import Data.Random.Source.Std
import Data.Random.Source.StdGen
import Control.Monad.State
import Data.Random hiding (shuffle)
import Data.ProtoLens (def)
import Proto.Tensorflow.Core.Framework.Tensor
    ( TensorProto
    , dtype
    , tensorShape
    )
import qualified Proto.Tensorflow.Core.Framework.TensorShape
  as TensorShape
import Text.Printf (printf)

-- For debugging only
import System.Random
import Debug.Trace


-- | "-1" means it can be arbitrary
minibatchSize    = -1    :: Int64
inputSize        = 200   :: Int64 
embeddedWordSize = 150   :: Int64
vocabSize        = 79951 :: Int64
convFeatures     = 1     :: Int64
convSize         = 3     :: Int64
stride           = 1     :: Int64
batchSize        = 100   :: Int
learningRate     = TF.scalar 1e-5


type WordIds    = TF.Tensor TF.Value Int32
type Probs      = TF.Tensor TF.Value Float


data Model      = Model {
      infer :: TF.TensorData Int32 -- word ids
            -> TF.Session (V.Vector Float)
    , train     :: TF.TensorData Int32 -> TF.TensorData Float -> TF.Session ()
    , loss      :: TF.TensorData Int32 -> TF.TensorData Float -> TF.Session Float
    , accuracy  :: TF.TensorData Int32 -> TF.TensorData Float -> TF.Session Float
    }


-- We need to build our own `conv2D` because the one in `GenOps` is broken.
conv2D :: forall v1 v2 t . (TF.TensorType t, 
                            TF.OneOf '[Data.Word.Word16, Double, Float] t)
          => TF.Tensor v1 t         -- ^ __input__
          -> TF.Tensor v2 t         -- ^ __filter__
          -> [Int64]                -- ^ __strides__
          -> TF.Tensor TF.Value t   -- ^ __output__
conv2D input filter strides | TF.eqLengthGuard [] =
    TF.buildOp (
             TF.opDef "Conv2D"
                & TF.opAttr "T"             .~ TF.tensorType (undefined :: t)
                & TF.opAttr "strides"       .~ strides
                & TF.opAttr "padding"       .~ ("SAME" :: ByteString)
                & TF.opAttr "data_format"   .~ ("NHWC" :: ByteString)
             )
        input filter


randomParam :: TF.Shape ->
               TF.Build (TF.Tensor TF.Value Float)
randomParam (TF.Shape shape) =
    (* stddev) <$> TF.truncatedNormal (TF.vector shape)
  where 
    stddev = TF.scalar 0.0001


reduceMean xs = TF.mean xs (TF.scalar (0 :: Int32))


-- | ...
createModel :: TF.Build Model
createModel = do
    -- Only "titles" for now; forget the masks and what-not.
    titles      <- TF.placeholder [minibatchSize, inputSize, 1] :: TF.Build WordIds
    probs       <- TF.placeholder [minibatchSize]               :: TF.Build Probs

    embedding    <- TF.initializedVariable =<< randomParam [vocabSize, embeddedWordSize]
    wordVectors  <- TF.embeddingLookup [embedding] titles


    -- Convolution
    convWeights <- TF.initializedVariable =<< randomParam [convSize, 1, embeddedWordSize, convFeatures]
    convBias    <- TF.zeroInitializedVariable [convFeatures]

    let conv'   = conv2D wordVectors convWeights [1, stride, 1, 1]
    let convOut = conv' `TF.add` convBias
        means   = TF.mean convOut (TF.vector [1,2,3 :: Int32])


    -- Loss
    sigmoidLoss <- TF.sigmoidCrossEntropyWithLogits means probs

    let finalLoss = reduceMean sigmoidLoss
        params    = [convWeights, convBias, embedding]

    
    -- Training (by Gradient Descent)
    
    grads <- TF.gradients finalLoss params

    let applyGrad param grad = TF.assign param $ param `TF.sub` (learningRate * grad)
    
    trainStep <- TF.group =<< zipWithM applyGrad params grads

    -- TODO: Implement Adam instead.


    -- Inference
    let finalProbs = TFC.sigmoid means
    predict  <- TF.render finalProbs


    -- Accuracy
    let rightEnough = TF.equal probs (TFC.cast (TFC.greater finalProbs (TF.scalar 0.5)))
        accuracyT   = reduceMean (TFC.cast rightEnough)


    return Model {
          train = \ts ps -> TF.runWithFeeds_ [
              TF.feed titles ts
            , TF.feed probs  ps
            ] trainStep

        , infer = \ts -> TF.runWithFeeds [TF.feed titles ts] predict

        , loss = \ts ps -> TF.unScalar <$> TF.runWithFeeds [
              TF.feed titles ts
            , TF.feed probs  ps
            ] finalLoss

        , accuracy = \ts ps -> TF.unScalar <$> TF.runWithFeeds [
              TF.feed titles ts
            , TF.feed probs  ps
            ] accuracyT
        }


data Article = Article {
      uid              :: !Text
    , titleIds         :: !Text
    , abstractIds      :: !Text
    , sciteProbability :: !Float
    } deriving (Show)


trainDataDir :: FilePath
trainDataDir = "../data/noon/"


readArticles :: FilePath -> IO (V.Vector Article)
readArticles dataset = do
    csvData <- BL.readFile (trainDataDir </> dataset <.> "csv") 
    case decodeByName csvData of
        Left err     -> error $ "Error reading csv: " ++ (show err)
        Right (_, v) -> return v


instance FromNamedRecord Article where
    parseNamedRecord r = Article <$> 
            r .: "id"
        <*> r .: "wordset_1_ids"
        <*> r .: "wordset_2_ids"
        <*> r .: "probability"


main = TF.runSession $ do
    liftIO $ putStrLn "Loading and shuffling data ..."

    trainArticles'      <- fmap V.toList $ liftIO (readArticles "train")
    validationArticles' <- fmap V.toList $ liftIO (readArticles "validation")

    trainArticles       <- liftIO (runRVar (shuffle trainArticles'     ) StdRandom :: IO [Article])
    validationArticles  <- liftIO (runRVar (shuffle validationArticles') StdRandom :: IO [Article])
     

    liftIO $ putStrLn "Building model ..."
    model <- TF.build createModel

    let -- TODO: Better yet, have some appropriate method of data
        -- selection here, including epoch's and what-not.
        selectBatch i xs = take batchSize $ drop (i * batchSize) (cycle xs)

    liftIO $ putStrLn "Training ..."
    forM_ ([0..10000] :: [Int]) $ \i -> do
        let batch              = selectBatch i trainArticles

            idStrToIds :: Article -> [Int32]
            idStrToIds article = fromIntegral . read . unpack <$> splitOn " " (titleIds article)

            imgDim             = [fromIntegral batchSize, inputSize, 1] :: TF.Shape
            probDim            = [fromIntegral batchSize]               :: TF.Shape

            -- Madness.
            -- flatten :: V.Vector [Int32] -> V.Vector Int32
            -- flatten vs = V.fromList $ join $ V.toList vs
            flatten = join

            -- We need to append zeros so that the numbers here aren't random
            -- indicies we look up in the embedding.
            
            wordIds = idStrToIds <$> batch
            wordIdsWithZeros = fmap (\xs -> xs ++ replicate (fromIntegral inputSize - length xs) 0) wordIds

            wordIdsFlat = V.fromList $ flatten wordIdsWithZeros
            sciteProbs  = V.fromList $ sciteProbability <$> batch

            titles = TF.encodeTensorData imgDim  wordIdsFlat :: TF.TensorData Int32
            probs  = TF.encodeTensorData probDim sciteProbs  :: TF.TensorData Float

        -- Train
        train model titles probs

        when (i `mod` 100 == 0) $ do

            modelLoss <- loss model titles probs
            accuracy  <- accuracy model titles probs

            liftIO $ do
                putStrLn ""
                putStrLn $ "         i: " ++ show i
                putStrLn $ "      loss: " ++ show modelLoss
                putStrLn $ "  accuracy: " ++ show accuracy

    liftIO $ putStrLn "Done"
